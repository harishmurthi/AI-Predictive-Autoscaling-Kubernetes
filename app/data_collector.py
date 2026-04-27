import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
from kubernetes import client, config

LOGGER = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    timestamp: float
    cpu_usage_percent: float
    memory_usage_percent: float
    request_rate_rps: float
    current_replicas: float


class PrometheusCollector:
    def __init__(
        self,
        prometheus_url: str,
        namespace: str,
        deployment_name: str,
        timeout: int = 10,
        verify_ssl: bool = True,
        cpu_rate_window: str = "30s",
        request_rate_window: str = "30s",
        request_rate_queries: Optional[List[str]] = None,
    ) -> None:
        self.base_url = prometheus_url.rstrip("/")
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.cpu_rate_window = cpu_rate_window
        self.request_rate_window = request_rate_window
        self.request_rate_queries = request_rate_queries or []
        self.session = requests.Session()
        self._apps_api: Optional[client.AppsV1Api] = None

    def _get_current_replicas(self) -> float:
        fallback_replicas = 1.0
        try:
            if self._apps_api is None:
                config.load_incluster_config()
                self._apps_api = client.AppsV1Api()

            deployment = self._apps_api.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace,
            )
            replicas = deployment.spec.replicas if deployment.spec and deployment.spec.replicas is not None else 1
            return float(max(1, int(replicas)))
        except Exception as exc:
            LOGGER.warning(
                "Failed to fetch deployment replicas for %s/%s; using fallback %.1f. Error: %s",
                self.namespace,
                self.deployment_name,
                fallback_replicas,
                exc,
            )
            return fallback_replicas

    def _query(self, promql: str) -> Optional[float]:
        url = f"{self.base_url}/api/v1/query"
        try:
            response = self.session.get(
                url,
                params={"query": promql},
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != "success":
                LOGGER.warning("Prometheus query unsuccessful: %s", payload)
                return None

            results = payload.get("data", {}).get("result", [])
            if not results:
                LOGGER.warning("No result for query: %s", promql)
                return None

            # Sum all vector values for robust aggregation when multiple pods are returned.
            total = 0.0
            for item in results:
                value = item.get("value")
                if not value or len(value) < 2:
                    continue
                total += float(value[1])
            return total
        except requests.RequestException as exc:
            LOGGER.exception("Prometheus request failed: %s", exc)
            return None
        except (ValueError, TypeError) as exc:
            LOGGER.exception("Failed to parse Prometheus response: %s", exc)
            return None

    def _query_first_available(self, named_queries: List[Tuple[str, str]]) -> Tuple[float, str]:
        for query_name, promql in named_queries:
            value = self._query(promql)
            if value is not None:
                return value, query_name
        return 0.0, "fallback_zero"

    def _build_queries(self) -> Dict[str, str]:
        pod_pattern = f"{self.deployment_name}.*"

        cpu_percent_query = (
            "100 * sum(rate(container_cpu_usage_seconds_total{"
            f'namespace="{self.namespace}",pod=~"{pod_pattern}",container!="POD",container!=""'
            f"}}[{self.cpu_rate_window}]))"
            " / "
            "sum(kube_pod_container_resource_limits{"
            f'namespace="{self.namespace}",pod=~"{pod_pattern}",resource="cpu",unit="core"'
            "})"
        )

        memory_percent_query = (
            "100 * sum(container_memory_working_set_bytes{"
            f'namespace="{self.namespace}",pod=~"{pod_pattern}",container!="POD",container!=""'
            "})"
            " / "
            "sum(kube_pod_container_resource_limits{"
            f'namespace="{self.namespace}",pod=~"{pod_pattern}",resource="memory",unit="byte"'
            "})"
        )

        request_rate_queries = [
            (
                "http_requests_total",
                "sum(rate(http_requests_total{"
                f'namespace="{self.namespace}",pod=~"{pod_pattern}"'
                f"}}[{self.request_rate_window}]))",
            ),
            (
                "nginx_http_requests_total",
                "sum(rate(nginx_http_requests_total{"
                f'namespace="{self.namespace}",pod=~"{pod_pattern}"'
                f"}}[{self.request_rate_window}]))",
            ),
            (
                "nginx_ingress_controller_requests",
                "sum(rate(nginx_ingress_controller_requests{"
                f'namespace="{self.namespace}"'
                f"}}[{self.request_rate_window}]))",
            ),
            (
                "network_receive_packets_proxy",
                "sum(rate(container_network_receive_packets_total{"
                f'namespace="{self.namespace}",pod=~"{pod_pattern}"'
                f"}}[{self.request_rate_window}]))",
            ),
            (
                "network_receive_bytes_proxy",
                "sum(rate(container_network_receive_bytes_total{"
                f'namespace="{self.namespace}",pod=~"{pod_pattern}"'
                f"}}[{self.request_rate_window}])) / 1024",
            ),
        ]

        for custom_query in self.request_rate_queries:
            request_rate_queries.insert(0, ("custom_request_rate_query", custom_query))

        return {
            "cpu": cpu_percent_query,
            "memory": memory_percent_query,
            "request_rate_candidates": request_rate_queries,
        }

    def collect(self) -> MetricSnapshot:
        queries = self._build_queries()

        cpu_usage = self._query(queries["cpu"]) or 0.0
        memory_usage = self._query(queries["memory"]) or 0.0
        # If request-rate query has no data, default to 0.0.
        request_rate, request_rate_source = self._query_first_available(queries["request_rate_candidates"])
        current_replicas = self._get_current_replicas()

        cpu_usage = max(0.0, min(cpu_usage, 100.0))
        memory_usage = max(0.0, min(memory_usage, 100.0))
        request_rate = max(0.0, request_rate)
        current_replicas = max(1.0, current_replicas)

        snapshot = MetricSnapshot(
            timestamp=time.time(),
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory_usage,
            request_rate_rps=request_rate,
            current_replicas=current_replicas,
        )
        LOGGER.info(
            "Collected metrics: cpu=%.2f%% memory=%.2f%% req_rate=%.3f source=%s replicas=%.0f",
            snapshot.cpu_usage_percent,
            snapshot.memory_usage_percent,
            snapshot.request_rate_rps,
            request_rate_source,
            snapshot.current_replicas,
        )
        return snapshot
