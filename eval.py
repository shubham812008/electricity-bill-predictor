"""Interactive evaluator for the electricity consumption model.

Run Model.py first to create electricity_model.pkl, then run:
    python eval.py

The model expects the previous seven daily consumption values in kWh.
Previous_Day_1 means yesterday; Previous_Day_7 means seven days ago.
"""
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / "electricity_model.pkl"
FEATURES = [f"Previous_Day_{day}" for day in range(1, 8)]


def read_non_negative_number(prompt: str) -> float:
    """Read a numeric value that must be zero or greater."""
    while True:
        raw_value = input(prompt).strip()
        try:
            value = float(raw_value)
        except ValueError:
            print("Please enter a valid number, for example: 12.5")
            continue

        if value < 0:
            print("Consumption and tariff cannot be negative. Try again.")
            continue
        return value


def main() -> None:
    if not MODEL_PATH.exists():
        print("Trained model not found:", MODEL_PATH.name)
        print("First put daily_electricity.csv beside Model.py and run:")
        print("    python Model.py")
        print("Then run this file with:")
        print("    python eval.py")
        return

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        print(f"Could not load {MODEL_PATH.name}: {exc}")
        print("Try training the model again by running: python Model.py")
        return

    print("\n=== Electricity Consumption Predictor ===")
    print("Enter daily electricity consumption in kWh for the previous 7 days.")
    print("Use yesterday first, then work backwards.\n")

    values = {}
    for day in range(1, 8):
        label = "yesterday" if day == 1 else f"{day} days ago"
        values[f"Previous_Day_{day}"] = read_non_negative_number(
            f"Consumption for {label} (kWh): "
        )

    # A DataFrame preserves the exact feature names and order used during training.
    input_data = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)

    try:
        prediction = float(model.predict(input_data)[0])
    except Exception as exc:
        print(f"Prediction failed: {exc}")
        print("Check that electricity_model.pkl was created by your current Model.py.")
        return

    if prediction < 0:
        print("\nNote: the model returned a negative value, which is not a valid")
        print("physical electricity consumption. Displaying 0.00 kWh instead.")
        prediction = 0.0

    print("\n----------- Prediction Result -----------")
    print(f"Predicted electricity consumption for the next day: {prediction:.2f} kWh")

    rate_text = input(
        "Enter your electricity rate per kWh (₹), or press Enter to skip bill estimate: "
    ).strip()
    if rate_text:
        try:
            rate = float(rate_text)
            if rate < 0:
                print("Tariff cannot be negative; bill estimate skipped.")
            else:
                estimated_bill = prediction * rate
                print(f"Estimated energy cost for that day: ₹{estimated_bill:.2f}")
                print("Estimate excludes fixed charges, taxes, and other fees.")
        except ValueError:
            print("Invalid tariff entered; bill estimate skipped.")

    print("\nPrediction completed.")


if __name__ == "__main__":
    main()
