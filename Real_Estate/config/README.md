# Config file

## Mandatory Parameters:

- purchase_price: The purchasing price of the property.
- rent: Monthly rent amount.
- surface: Surface area in square meters.
- tp_bien: Type of property: "ancien" for old property, "neuf" for new property.
- r_bien: Annual return of the real estate price.
- r_banque: Interest rate of the loan.
- r_assurance: Interest rate of the insurance to subscribe in case of disability or death.
- r_impot: Tax rate applied on the rent.
- r_plus_value: Tax rate when selling the real estate.
- r_prelevement_sociaux: Social deduction rate.

## Optional Parameters:

- start_date_achat: Starting date of the acquisition in format "YYYY-MM-DD".
- start_date_loyer: Starting date of the first rent collected in format "YYYY-MM-DD".
- T: Maturity of the loan in years.
- T_differe: Number of years during which only the interest on the loan is paid, not the principal. 'T'-'T_differe' represents the number of years where both interest and principal are paid.
- travaux: Cost of construction work.
- pret_travaux: True or False, whether the construction work cost is included in the loan.
- meubles: Furnishing cost.
- pret_meubles: True or False, whether the furnishing cost is included in the loan.
- taux_apport: Percentage of the loan to be paid out of pocket, excluding notary fees.
- r_loyer: Annual rate of return of the rent.
- frais_notaire: Notary fees.
- frais_prorata: Annual fees already paid by the seller to be reimbursed (property tax, co-ownership fees, etc.).
- frais_signature: Signature fees.
- frais_agence_achat: Agency fees upon purchase.
- frais_agence_vente: Agency fees upon sale.
- frais_garantis: Loan guarantee fees in case of failure to pay.
- frais_dossier: Processing fee for the loan.
- frais_assurance_emprunt: Processing fee for the insurance.
- frais_comptable: Accounting fees (annual).
- frais_gestion: Percentage of the rent as a management fee.
- frais_service: Service charges for gas, electricity, WiFi, etc. (monthly).
- assurance_GLI: Unpaid rent guarantee as a percentage of the monthly rent.
- assurance_habitation: Home insurance to protect against property risk (annual).
- copropriete: Co-ownership fees (quarterly).
- amortissement: Depreciable expenses ("structure", "IGT", "agencement", "facade", "travaux", "meubles").

    key: String, name of the charge.
    value: Dictionary with 'montant' (total amount of the charge) and 'duree' (number of years amortizable).

- r_innocupation: Vacancy rate (frequency).
- r_sinistre: Loss rate (frequency).
- frais_innocupation: Additional management fees to find a new tenant.
- frais_sinistre: Average cost when a loss occurs.

For more details on the default parameters, see the '../LMNP.py' file.