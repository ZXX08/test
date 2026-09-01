import importlib
import sys


EXPECTED = {
    "torch": "2.13.0+cu130",
    "transformers": "5.2.0",
    "datasets": "4.0.0",
    "accelerate": "1.11.0",
    "peft": "0.18.1",
    "trl": "0.24.0",
    "deepspeed": "0.18.4",
    "av": "16.0.0",
    "decord": "0.6.0",
    "pyarrow": "25.0.1",
    "llamafactory": "0.9.5.dev0",
}


def main():
    print("Python:", sys.version.replace("\n", " "))
    failures = []
    for package, expected in EXPECTED.items():
        try:
            module = importlib.import_module(package)
            actual = getattr(module, "__version__", None)
            status = "OK" if actual == expected else "MISMATCH"
            print(f"{package}: {actual} [{status}; expected {expected}]")
            if actual != expected:
                failures.append(f"{package}: {actual} != {expected}")
        except Exception as exc:
            print(f"{package}: import failed: {exc}")
            failures.append(f"{package}: import failed")

    import torch

    print("Compiled CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("Visible GPUs:", torch.cuda.device_count())
    if failures:
        raise SystemExit("Environment verification failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()

