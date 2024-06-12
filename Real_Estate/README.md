# Real Estate Investment 

This project aims to determine the profitability of your real estate investment under the LMNP (Loueur Meublé Non-Professionnel) regime in France. This tool is useful for comparing and identifying the most profitable real estate investments. Several metrics, each with their own quality ranges, are calculated to evaluate your investment's performance:

- Annual Cashflow: Can be positive or negative depending on expenses (taxes, co-ownership fees, management fees, loan financing, notary fees, etc.) and income (collected rents).
- Percentage of Property Paid: The percentage of the property cost that you have paid out of pocket, with the remainder covered by rental income.
- Gross Return: Calculated as annual_rent / purchase_price.
- Cap Rate: Calculated as (annual_rent - expenses) / purchase_price.
- Net Return: Calculated as (annual_rent - expenses - interest) / purchase_price.
- Net-Net Return: Calculated as (annual_rent - expenses - interest - taxes) / purchase_price.
- Internal Rate of Return (IRR): The discounted rate r at which all future cash flows replicate the initial investment. This metric is useful for comparing real estate investments against other financial investments.

## LMNP.py

This is the main file for defining a real estate investment. The code is organized into six classes:

- Notary: Handles notary fees paid upfront.
- Financing: Manages loan financing.
- Property: Includes management fees, rent, furnishing, and construction work.
- Taxation: Manages real estate taxes.
- Investment: Computes the metrics defined above.
- MC: A Monte Carlo class built on top of Investment to incorporate random costs (e.g., vacancy rates, loss costs).

## config

This folder contains configuration files that set the parameters for running your investment analysis.

## run.ipynb

This Jupyter notebook forecasts the performance of your real estate investment.