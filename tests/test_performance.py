import pytest
import time
import requests
import os

API_BASE = "http://localhost:8000"

def test_api_latency():
    """Verify that the API status endpoint responds within a reasonable time."""
    try:
        start_time = time.time()
        response = requests.get(f"{API_BASE}/api/status", timeout=2)
        latency = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        assert latency < 500  # API should respond in < 500ms
        print(f"API Latency: {latency:.2f}ms")
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")

def test_static_css_exists():
    """Verify that the static CSS file is present."""
    css_path = os.path.join(os.getcwd(), "static", "style.css")
    assert os.path.exists(css_path)

def test_manual_exists():
    """Verify that the manual markdown file is present."""
    manual_path = os.path.join(os.getcwd(), "docs", "manual.md")
    assert os.path.exists(manual_path)

if __name__ == "__main__":
    pytest.main([__file__])
