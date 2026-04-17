"""
Test fixture script for REST API server.
ML Principle: Evaluates live forecasting throughput and API data serialization consistency.
"""
import requests, json, time

start = time.time()
r = requests.get('http://localhost:8000/api/live-forecast?date=2026-04-13', timeout=30)
elapsed = time.time() - start

print(f"Status: {r.status_code}")
print(f"Time: {elapsed:.1f}s")
d = r.json()
print(f"Date: {d['date']}")
print(f"Source: {d['data_source']}")
print(f"Avg Price: {d['avg_price']}")
print(f"Cheapest: {d['cheapest_hour']:02d}:00")
print(f"Prices (first 6): {d['predicted_price'][:6]}")
print(f"Load (first 6): {d['predicted_consumption'][:6]}")
print(f"Ensemble: {json.dumps(d['ensemble_config'], indent=2)}")
