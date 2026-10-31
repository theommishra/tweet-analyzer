import os
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from model import EmotionClassifier, EKMAN_EMOTIONS

# Global instance of classifier
classifier = EmotionClassifier()

class EmotionRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler serving static frontend assets and REST API endpoints.
    """

    def log_message(self, format, *args):
        # Clean logging format
        print(f"[{self.log_date_time_string()}] {args[0]}")

    def _set_json_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _set_file_headers(self, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_json_headers(200)

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        static_dir = os.path.join(os.path.dirname(__file__), "static")

        if path == "/api/health":
            self._set_json_headers(200)
            response = {
                "status": "healthy",
                "model": "Ekman-7-Classifier",
                "model_version": getattr(classifier, "model_version", "unknown"),
                "taxonomy": EKMAN_EMOTIONS,
                "backend": classifier.backend,
                "cv_macro_f1": getattr(classifier, "cv_macro_f1", None)
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return

        # Static file routing
        if path == "/" or path == "/index.html":
            file_path = os.path.join(static_dir, "index.html")
            mime_type = "text/html"
        elif path in ["/style.css", "/static/style.css"]:
            file_path = os.path.join(static_dir, "style.css")
            mime_type = "text/css"
        elif path in ["/app.js", "/static/app.js"]:
            file_path = os.path.join(static_dir, "app.js")
            mime_type = "application/javascript"
        else:
            file_path = os.path.join(static_dir, path.lstrip("/"))
            if path.endswith(".html"):
                mime_type = "text/html"
            elif path.endswith(".css"):
                mime_type = "text/css"
            elif path.endswith(".js"):
                mime_type = "application/javascript"
            elif path.endswith(".json"):
                mime_type = "application/json"
            else:
                mime_type = "text/plain"

        if os.path.exists(file_path) and os.path.isfile(file_path):
            self._set_file_headers(mime_type, 200)
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self._set_json_headers(404)
            self.wfile.write(json.dumps({"error": f"Path '{path}' not found."}).encode("utf-8"))

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length)

        try:
            data = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception as e:
            self._set_json_headers(400)
            self.wfile.write(json.dumps({"error": f"Invalid JSON body: {e}"}).encode("utf-8"))
            return

        if path == "/api/predict":
            text = data.get("text", "")
            result = classifier.predict(text)
            self._set_json_headers(200)
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        elif path == "/api/preprocess":
            text = data.get("text", "")
            details = classifier.preprocessor.get_preprocessing_details(text)
            self._set_json_headers(200)
            self.wfile.write(json.dumps(details).encode("utf-8"))
            return

        else:
            self._set_json_headers(404)
            self.wfile.write(json.dumps({"error": f"Endpoint '{path}' not found."}).encode("utf-8"))

def run_server(port=8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, EmotionRequestHandler)
    print(f"==================================================")
    print(f" Ekman Emotion Detection Server Running!")
    print(f" URL: http://localhost:{port}")
    print(f" Model Engine: Scikit-Learn TF-IDF (100% Local)")
    print(f" Taxonomy: {', '.join(EKMAN_EMOTIONS)}")
    print(f"==================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully.")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
