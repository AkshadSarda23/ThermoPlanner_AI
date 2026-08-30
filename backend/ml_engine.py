import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class ThermalROIEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self._fit_synthetic_baseline()

    def _fit_synthetic_baseline(self):
        np.random.seed(42)
        X = np.random.uniform(low=[32.0, 28.0, 0.3], high=[52.0, 42.0, 0.95], size=(300, 3))
        y = (X[:, 0] * 18.5) + (X[:, 1] * 8.2) + (X[:, 2] * 420.0) + np.random.normal(0, 15.0, 300)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def evaluate_zone(self, surface_temp: float, air_temp: float, density: float):
        features = np.array([[surface_temp, air_temp, density]])
        features_scaled = self.scaler.transform(features)
        annual_hvac_surge = round(float(self.model.predict(features_scaled)[0]), 2)
        
        if surface_temp > 44.0:
            intervention = "High-Albedo Cool Roof Coating (SRI > 82) + Urban Tree Canopy"
            temp_drop = round(float(surface_temp * 0.12), 2)
            efficiency_gain = 0.38
        elif surface_temp > 38.0:
            intervention = "Reflective Solar Pavement & Roof Coating"
            temp_drop = round(float(surface_temp * 0.08), 2)
            efficiency_gain = 0.25
        else:
            intervention = "Targeted Vertical Vegetation Walls"
            temp_drop = round(float(surface_temp * 0.04), 2)
            efficiency_gain = 0.14

        projected_savings = round(annual_hvac_surge * efficiency_gain, 2)
        capex_estimate = round(density * 3500.0, 2)
        payback_months = round((capex_estimate / (projected_savings + 1e-5)) * 12, 1)
        co2_offset_tons = round(projected_savings * 0.00042, 2)

        return {
            "hvac_surge_usd": annual_hvac_surge,
            "recommended_action": intervention,
            "projected_temp_reduction_c": temp_drop,
            "annual_financial_savings_usd": projected_savings,
            "estimated_capex_usd": capex_estimate,
            "payback_period_months": payback_months,
            "annual_co2_offset_tons": co2_offset_tons
        }