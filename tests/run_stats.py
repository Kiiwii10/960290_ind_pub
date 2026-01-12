import argparse
import json
from urllib.request import Request, urlopen

def http_stats_get(url: str, timeout: int = 60) -> dict:
    req = Request(url, headers={"Content-Type": "application/json"}, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
    
def main():
    parser = argparse.ArgumentParser(description="Run RAG API tests against local server")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL of the RAG API server")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print(f"Fetching stats from {base_url}/api/stats")
    stats = http_stats_get(f"{base_url}/api/stats")
    print("Stats:", stats)


if __name__ == "__main__":
    main()