import requests
import sys

def test_server():
    try:
        print("Checking index page...")
        r = requests.get('http://127.0.0.1:5000/')
        print(f"Index status: {r.status_code}")
        
        print("\nChecking generator endpoint...")
        r = requests.get('http://127.0.0.1:5000/generate')
        print(f"Generator status: {r.status_code}")
        print(f"Content-type: {r.headers.get('Content-Type')}")
        print(f"Content length: {len(r.content)} bytes")
        
        if len(r.content) > 100:
            print("Successfully received image data.")
        else:
            print("Received empty or very small response.")
            
        print("\nChecking status endpoint...")
        r = requests.get('http://127.0.0.1:5000/status')
        print(f"Status status: {r.status_code}")
        print(f"Response: {r.json()}")
        
    except Exception as e:
        print(f"Server check failed: {e}")

if __name__ == "__main__":
    test_server()
