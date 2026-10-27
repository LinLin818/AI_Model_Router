import time
from router import Router

r = Router.load("artifacts/router.joblib")
prompts = ["What is the capital of France?"] * 1000
r.route(prompts[0])  # warm up
t = time.perf_counter()
for p in prompts:
    r.route(p)
print(f"{(time.perf_counter() - t) / len(prompts) * 1000:.2f} ms per prompt")