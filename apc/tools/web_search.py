"""Simulated web search tool for demo purposes."""

from __future__ import annotations

from typing import Any

from .base import Tool

_CANNED_RESULTS: dict[str, str] = {
    # --- Physics & Math constants ---
    "speed of light": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "gravitational constant": "The gravitational constant G is approximately 6.674 × 10^-11 N⋅m²/kg².",
    "planck constant": "The Planck constant h is approximately 6.626 × 10^-34 J⋅s.",
    "avogadro number": "Avogadro's number is approximately 6.022 × 10^23 mol^-1.",
    "pi digits": "Pi (π) = 3.14159265358979323846...",
    "euler number": "Euler's number (e) = 2.71828182845904523536...",
    "boiling point of water": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.",
    "distance earth to sun": "The average distance from Earth to the Sun is about 149.6 million kilometers (1 AU).",
    "distance earth to moon": "The average distance from Earth to the Moon is about 384,400 kilometers.",
    # --- Geography & Demographics ---
    "population of earth": "The world population is approximately 8.1 billion people as of 2024.",
    "population of india": "The population of India is approximately 1.44 billion people as of 2024.",
    "population of united states": "The population of the United States is approximately 334 million people as of 2024.",
    "population of china": "The population of China is approximately 1.41 billion people as of 2024.",
    "area of united states": "The total area of the United States is approximately 9,833,520 square kilometers (3,796,742 square miles).",
    "area of india": "The total area of India is approximately 3,287,263 square kilometers (1,269,219 square miles).",
    "area of russia": "The total area of Russia is approximately 17,098,242 square kilometers (6,601,670 square miles).",
    "height of mount everest": "Mount Everest is 8,849 meters (29,032 feet) tall, the highest point on Earth.",
    "depth of mariana trench": "The Mariana Trench reaches a maximum depth of about 10,994 meters (36,070 feet).",
    "length of amazon river": "The Amazon River is approximately 6,400 kilometers (3,976 miles) long.",
    "length of nile river": "The Nile River is approximately 6,650 kilometers (4,130 miles) long.",
    # --- Finance & Economics ---
    "usd to eur exchange rate": "The current USD to EUR exchange rate is approximately 0.92 (1 USD = 0.92 EUR).",
    "usd to gbp exchange rate": "The current USD to GBP exchange rate is approximately 0.79 (1 USD = 0.79 GBP).",
    "usd to jpy exchange rate": "The current USD to JPY exchange rate is approximately 149.50 (1 USD = 149.50 JPY).",
    "usd to inr exchange rate": "The current USD to INR exchange rate is approximately 83.10 (1 USD = 83.10 INR).",
    "gold price per ounce": "The current price of gold is approximately $2,340 per troy ounce.",
    "bitcoin price": "The current price of Bitcoin is approximately $67,500 USD.",
    "s&p 500 index": "The S&P 500 index is currently at approximately 5,450 points.",
    "us federal interest rate": "The US federal funds rate is currently 5.25% to 5.50%.",
    "us inflation rate": "The US annual inflation rate is approximately 3.4% as of 2024.",
    # --- Health & Nutrition ---
    "calories in banana": "A medium banana contains approximately 105 calories, 27g carbs, 1.3g protein, and 0.4g fat.",
    "calories in egg": "A large egg contains approximately 72 calories, 6.3g protein, 4.8g fat, and 0.4g carbs.",
    "calories in rice": "One cup of cooked white rice contains approximately 206 calories and 45g carbs.",
    "daily calorie intake": "The recommended daily calorie intake is about 2,000 calories for women and 2,500 for men.",
    "normal body temperature": "Normal human body temperature is approximately 37°C (98.6°F).",
    "average human heart rate": "The average resting heart rate for adults is 60 to 100 beats per minute.",
    "recommended water intake": "The recommended daily water intake is about 3.7 liters for men and 2.7 liters for women.",
    "bmi formula": "BMI = weight (kg) / height (m)^2. Healthy range is 18.5 to 24.9.",
    # --- Everyday / General knowledge ---
    "miles in kilometer": "1 kilometer equals approximately 0.6214 miles. 1 mile equals approximately 1.6093 kilometers.",
    "pounds in kilogram": "1 kilogram equals approximately 2.2046 pounds. 1 pound equals approximately 0.4536 kilograms.",
    "liters in gallon": "1 US gallon equals approximately 3.7854 liters. 1 liter equals approximately 0.2642 US gallons.",
    "us minimum wage": "The US federal minimum wage is $7.25 per hour as of 2024.",
    "us sales tax": "US sales tax varies by state, ranging from 0% (Oregon, Montana) to 7.25% (California). The average is about 5.09%.",
    "tip percentage": "In the United States, the customary tip for restaurant service is 15% to 20% of the pre-tax bill.",
}


class WebSearchTool(Tool):
    """Simulated web search that returns canned results for demo purposes."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for factual information. Returns relevant snippets."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "").lower().strip()
        if not query:
            return "Error: no query provided"

        for key, result in _CANNED_RESULTS.items():
            if key in query or query in key:
                return result

        return f"No results found for '{query}'. Try a different search term."
