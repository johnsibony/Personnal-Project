# Ajouter taxe fonciere

import pandas as pd
from pandas.tseries.offsets import DateOffset
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar
import copy
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Callable
import inspect
from joblib import Parallel, delayed

def filtre_dict(func: Callable, d: Dict) -> Dict:
    """
    Retourne le sous dicionnaire de 'd' matchant les arguments de 'func'.
    
    Parametres
    ----------
    func : Callable
        Fonction avec les arguments d'intérêts.
    d : dict
        dictionnaire incluant des pairs (key, value) en trop.
    """
    func_params = inspect.signature(func).parameters
    filtered_d = {k: v for k, v in d.items() if k in func_params}
    return filtered_d

@dataclass
class Notaire:
    """
    Une class modélisant les frais notariale.
    
    Attributs
    ----------
    prix_achat : float
        Prix d'achat.
    travaux : float
        Prix des travaux.
    meubles : float
        Prix d'ammeublement.
    frais_notaire : float
        Frais du notaire.
    frais_agence_achat : float
        Frais d'agence à l'achat.
    frais_prorata : float
        Frais payés à l'avance par le vendeur à rembourser.
    frais_signature : float
        Frais de signature.
    """
        
    prix_achat: float
    travaux: float
    meubles: float
    frais_notaire: float
    frais_agence_achat: float
    frais_prorata: float
    frais_signature: float

@dataclass
class Financement:
    """
    Une class modélisant le prêt.
    
    Le prêt doit obligatoirement être assuré par une assurance d'emprunt en cas de décès ou invalidité.
    Cette assurance est payés trimestriellement auquel s'additionne un coût fixe (frais d'assurance emprunt).

    De plus, il doit être couvert par un organisme de caution (Credit Logement en général)
    en cas de défaut de paiement. Cette caution (frais de garentis) est payée en une fois est
    comprend une part de comission pour rénumérer l'organisme et une part de caution appelé fond de mutuel.
    
    Attributs
    ----------
    start_date : str
        Date de commencement.
    N : float
        Notionel: montant emprunté.
        En input, 'N' doit etre avant apport.
        En attribut, 'N' deviendra après apport (voir method '__post_init__')
    taux_apport : float
        Taux d'apport.
    apport : float.
        Somme apporté réduisant le montant d'emprunt.
        Ce paramètre n'est pas un input.
    T : float
        Durée total du prêt (en année).
    T_differe : float
        Durée du prêt en différé (en année).
        Durant cette période, l'emprunteur paye les intérets sans rembourser le capital.
        ('T' - 'T_differe') représente le nombre d'années en prêt classique.
        
    Notes
    -----
    L'attribut 'apport' n'est pas un input.
    Il est calculé automatiquement par la méthode '__post_init__'.
    """

    start_date: str
    N: float
    taux_apport: float
    apport: float = field(init=False)
    T: float
    T_differe: float
        
    def __post_init__(self) -> None:
        """
        Calcul 'apport' et modifie 'N' en lui soustrayant le montant d'apport.
        """
        if self.T_differe > self.T:
            raise ValueError("'T_differe' soit être inférieur à 'T'.")
        self.apport = self.N * self.taux_apport
        self.N *= (1 - self.taux_apport)
        
    def calcul_interet(self, r: float, frais_garantis: float, frais_dossier: float) -> pd.DataFrame:
        """
        Calcul les mensualités du pret (intérets + capital).
        
        Parametres
        ----------
        r : float
            Taux d'intéret du prêt.
        frais_garantis : float
            Frais de la garantis d'emprunt.
        frais_dossier : float
            Frais de dossier bancaire.
        """
        def init_df() -> pd.DataFrame:
            """
            Initialize the Dataframe.
            """
            end_year = pd.date_range(start=self.start_date, periods=12*40, freq='MS')[-1].year + 1
            indexes = pd.date_range(start=self.start_date, end=f"{end_year}-12-01", freq='MS')
            df = pd.DataFrame(index=indexes, columns=["interet", "capital", "capital_restant", "frais_garantis", "frais_dossier",
                                                      "apport_pret"])
            df.loc[df.index[0], "frais_garantis"] = frais_garantis
            df.loc[df.index[0], "frais_dossier"] = frais_dossier
            df.loc[df.index[0], "apport_pret"] = self.apport
            return df
        
        df = init_df()
        self.__calcul_interet_differe(df, r)
        self.__calcul_interet_post_differe(df, r)
        df.fillna(0, inplace=True)
        return df

    def __calcul_interet_differe(self, df: pd.DataFrame, r: float) -> None:
        """
        Calcul les intérets du pret sur la période différé [0, 'T_differe'].
        
        Parametres
        ----------
        df : DataFrame
            DataFrame initialisé durant la méthode 'calcul_interet'.
        r : float
            Taux d'intéret du prêt.
        """
        indexes = pd.date_range(start=self.start_date, periods=12*self.T_differe, freq='MS')
        df.loc[indexes, "interet"] = self.N * r / 12
        df.loc[indexes, "capital_restant"] = self.N

    def __calcul_interet_post_differe(self, df: pd.DataFrame, r: float) -> None:
        """
        Calcul les intérets du pret après la période différé ['T_differe', 'T'-'T_differe'].
        
        Parametres
        ----------
        df : DataFrame
            DataFrame initialisé durant la méthode 'calcul_interet'.
        r : float
            Taux d'intéret du prêt.
        """
        mensualites = (r / 12 * self.N) / (1 - (1 + r / 12)**-((self.T - self.T_differe) * 12))
        t = np.arange(1, 12 * (self.T - self.T_differe) + 1)
        capital_restant = (1 + r / 12)**t * (self.N - 12 * mensualites / r) + 12 * mensualites / r
        start_date = df.index[12 * self.T_differe]
        indexes = pd.date_range(start=start_date, periods=12*(self.T-self.T_differe), freq='MS')
        df.loc[indexes, "capital_restant"] = capital_restant
        df["capital"] = df["capital_restant"].diff().abs().fillna(0)
        df.loc[indexes, "interet"] = mensualites - df.loc[indexes, "capital"]
        
    def calcul_assurance(self, r: float, frais_assurance_emprunt: float, capital_restant: pd.Series) -> pd.DataFrame:
        """
        Calcul les primes mensuelles de l'assurance emprunt sur toute la période [O, 'T'].
        Le taux d'assurance TAEA est appliqué sur le capital restant au début de l'année prise en compte.
        
        Parametres
        ----------
        r : float
            Taux d'assurance TAEA (annuel).
        frais_assurance_emprunt : float
            Frais de l'assurance emprunt
        capital_restant : pd.Series
            Serie du capital restant calculé par la métode 'calcul_interet'.
        """
    
        def init_df() -> pd.DataFrame:
            end_year = pd.date_range(start=self.start_date, periods=12*40, freq='MS')[-1].year + 1
            indexes = pd.date_range(start=self.start_date, end=f"{end_year}-12-01", freq='MS')
            df = pd.DataFrame(index=indexes, columns=["prime_assurance", "capital_restant", "frais_assurance_emprunt"])
            df.loc[df.index[0], "frais_assurance_emprunt"] = frais_assurance_emprunt
            df["capital_restant"] = capital_restant
            return df
        
        df = init_df()
        df["previous_capital_restant"] = df["capital_restant"].shift().fillna(self.N)
        indexes = pd.date_range(start=df.index[0], periods=self.T, freq=DateOffset(years=1))
        df.loc[indexes, "prime_assurance"] = df.loc[indexes, "previous_capital_restant"] * r / 12
        indexes = pd.date_range(start=df.index[0], periods=12*self.T, freq=DateOffset(months=1))
        df.loc[indexes, "prime_assurance"] = df.loc[indexes, "prime_assurance"].fillna(method="ffill")
        df.fillna(0, inplace=True)
        df.drop(columns=["capital_restant", "previous_capital_restant"], inplace=True)
        return df
    
@dataclass
class BienImmobilier:
    """
    Une class modélisant les profits et frais d'un bien.
    
    Attributs
    ----------
    start_date_achat : str
        Date de commencement du prêt.
    start_date_loyer : str
        Date du premier loyer percu.
    loyer : float
        loyer (mensuel).
    cout_travaux : float.
        Cout des travaux non financé par le prêt.
    cout_meubles : float
        Cout d'ammeublement non financé par le prêt.
        
    Notes
    -----
    'cout_travaux' et 'cout_meubles' sont les montants payés de la poche de l'investisseur et non par le prêt.
    Si ces cout sont financés par le pret, ils doivent être mis à zeros ici. Sinon, ils seront comptés double.
    """
        
    start_date_achat: str
    start_date_loyer: str
    loyer: float
    cout_travaux: float
    cout_meubles: float
        
    def calcul_frais(self, r_loyer: float, frais_gestion: float, assurance_GLI: float, assurance_habitation: float, copropriete: float, frais_service: float) -> pd.DataFrame:
        """
        Défini un DataFrame regroupant les cashflow des frais et revenus.

        Parametres
        ----------
        r_loyer : float
            Taux annuel d'accroisement des loyers.
        frais_gestion : float
            Taux retranché aux loyers pour payer l'agence de gestion.
        assurance_GLI : float
            Assurance garantis de loyer impayés. Taux retranché aux loyers pour payer l'assurance.
            En général cette assurance est nulle (soit payé par le locataire, soit le locataire possède un garant).
        assurance_habitation : float
            Frais (annuel) d'assurance pour le bien contre les sinistres.
        copropriete: float
            Frais de copropriété (trimestriel) pour les charges courantes de l'immeuble. Des appels de roulement sont
            possible pour réapprovisionner le fond de trésorerie si utilisé. Un fond de réserve pour la réalisation de futures
            gros travaux peut également être demander.
        frais_service: float
            Frais (mensuel) de service tel que wifi, electricité, gaz.
            En général, ces charges sont payés par le locataire.
        """
        def init_df() -> pd.DataFrame:
            """
            Initialisation du DataFrame.
            """
            end_year = pd.date_range(start=self.start_date_achat, periods=12*40, freq='MS')[-1].year + 1
            indexes = pd.date_range(start=self.start_date_achat, end=f"{end_year}-12-01", freq='MS')
            df = pd.DataFrame(0, index=indexes, columns=["loyer", "frais_gestion", "assurance_GLI", "assurance_habitation",
                                                         "copropriete", "frais_service", "frais_sinistre", "cout_travaux",
                                                         "cout_meubles"])
            df.loc[df.index[0], "cout_travaux"] = self.cout_travaux
            df.loc[df.index[0], "cout_meubles"] = self.cout_meubles
            return df
        
        df = init_df()
        indexes = pd.date_range(start=df.index[0], end=df.index[-1], freq='QS')
        df.loc[indexes, "copropriete"] = copropriete
        indexes = pd.date_range(start=df.index[0], end=df.index[-1], freq=DateOffset(years=1))
        df.loc[indexes, "assurance_habitation"] = assurance_habitation
        df.loc[indexes, "rendement_loyer"] = [(1 + r_loyer)**t for t in range(len(indexes))]
        df["rendement_loyer"] = df["rendement_loyer"].fillna(method="ffill")
        df.loc[df.index > self.start_date_loyer, "frais_service"] = frais_service
        df.loc[df.index > self.start_date_loyer, "loyer"] = self.loyer
        df["loyer"] = df["loyer"] * df["rendement_loyer"]
        df["frais_gestion"] = df["loyer"] * frais_gestion
        df["assurance_GLI"] = df["loyer"] * assurance_GLI
        df.drop(columns=["rendement_loyer"], inplace=True)
        return df

@dataclass
class Fiscalite: ############### ADD TAXE_FONCIERE + CFE ###############
    """
    Une class modélisant l'imposition sur les revenus.
    
    Attributs
    ----------
    r : float
        Taux d'imposition.
        r = r1 + r2
        avec: - r1: tranche d'imposition de l'investisseur.
              - r2: prélèvements sociaux si résidents fiscaux francais OU prélèvements solidarité sinon.
    frais_comptable : float
        Frais comptable (annuel).
    notaire : Notaire
        Object de la classe Notaire.
    amortissement : Dict.
        Charges amortissables: "structure", "IGT", "agencement", "facade", "travaux", "meubles".
        key: str, nom de la charge.
        value: dict, ('montant': montant total de la charge, 'duree': nombre d'années amortissable).
    """
    r: float
    frais_comptable: float
    notaire: Notaire
    amortissement: Optional[Dict[str, Dict[str, float]]] = None
            
    def __post_init__(self) -> None:
        """
        Défini 'amortissement' si non spécifié.
        """
        prix_achat = self.notaire.prix_achat + self.notaire.frais_notaire + self.notaire.frais_agence_achat
        if self.amortissement is None:
            self.amortissement = {}
        self.amortissement.setdefault("structure", {"montant": 0.30 * prix_achat, "duree":50})
        self.amortissement.setdefault("IGT", {"montant": 0.15 * prix_achat, "duree": 15})
        self.amortissement.setdefault("agencement", {"montant": 0.20 * prix_achat, "duree": 5})
        self.amortissement.setdefault("facade", {"montant": 0.10 * prix_achat, "duree": 20})
        self.amortissement.setdefault("travaux", {"montant": self.notaire.travaux, "duree": 10})
        self.amortissement.setdefault("meubles", {"montant": self.notaire.meubles, "duree": 5})
        
    def __calcul_amortissement(self, indexes: pd.DatetimeIndex) -> pd.Series:
        """
        Calcul les amortissements annuel.
        
        Parametres
        ----------
        indexes : pd.DatetimeIndex
            dates ()
        """
        amortissement = pd.Series(0, index=indexes)
        prorata = (pd.Timestamp(year=amortissement.index[0].year, month=12, day=31) - amortissement.index[0]).days / 360
        for k, v in self.amortissement.items():
            amortissement.loc[amortissement.index[0]] += (v["montant"] / v["duree"]) * prorata
            indexes = pd.date_range(start=amortissement.index[0]+relativedelta(years=1),
                                    periods=v["duree"]-1,
                                    freq=DateOffset(years=1))
            amortissement.loc[set(indexes) & set(amortissement.index)] += (v["montant"] / v["duree"])
            last_date = amortissement.index[0] + relativedelta(years=v["duree"])
            amortissement.loc[set([last_date]) & set(amortissement.index)] += (v["montant"] / v["duree"]) * (1 - prorata)
        return amortissement
        
    def calcul_impot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcul l'imposition annuel sur les revenus selon les régime:
        
        - Regime micro bic: imposition sur 50% des revenus.
        
        - Regime réel simplifié: imposition sur (revenus - charges)
            avec charges = charges déductibles + charges amortissables à l'année.
            
        Le régime optimal est choisi chaque année pour calculer l'imposition minimal.
        Le régime micro bic est preferable lorsque les charges sont inférieur à 50% des revenus.
        
        Notes
        -----
        En général, le régime réel simplifié est préférable au début du prêt car les charges d'amortissement sont
        élevées et supérieurs à 50% des revenus. Puis, ces charges s'amoindrissent et le régime micro bic
        devient préférable.
        En général, le changement de régime réel simplifié - micro bic ne se réalise qu'une fois (ou jamais)
        au cours de l'investissement.
        
        Parametres
        ----------
        df : Dataframe résultant du merge entre les DataFrame résultant de 'Financement' et 'BienImmobilier'.
        """
        def preprocess(df: pd.DataFrame) -> pd.DataFrame:
            """
            Preprocess le DataFrame pour extraire seul les charges et revenus.
            
            Parametres
            ----------
            df : Dataframe résultant du merge entre les DataFrame résultant de 'Financement' et 'BienImmobilier'.
            """
            df = df.loc[:, ["loyer", "copropriete", "interet", "prime_assurance", "frais_garantis", "frais_dossier", "frais_assurance_emprunt", "assurance_habitation", "assurance_GLI", "frais_service", "frais_sinistre", "frais_gestion"]]
            df.loc[df.index[0], "frais_prorata"] = self.notaire.frais_prorata
            df["amortissement"] = self.__calcul_amortissement(df.index)
            start = df[df["loyer"] > 0].index[0]
            indexes = pd.date_range(start=start, end=df.index[-1], freq=DateOffset(years=1))
            df.loc[indexes, "frais_comptable"] = self.frais_comptable
            return df
        
        df = preprocess(df)
        loyer_annuel = df.groupby(df.index.year)["loyer"].sum()
        charges = df.groupby(df.index.year)[list(set(df.columns) - set(["loyer"]))].sum().sum(axis=1)
        reel_simplifie = loyer_annuel - charges
        reel_simplifie *= self.r
        reel_simplifie = reel_simplifie.apply(lambda x: max(x, 0))
        micro_bic = loyer_annuel * 0.5 * self.r
        
        first_year, last_year = df.index[0].year, df.index[-1].year
        indexes = pd.date_range(start=f"{first_year}-12-01", end=f"{last_year}-12-01", freq=DateOffset(years=1))
        loyer_condition = loyer_annuel.values <= 72600
        regime_condition = micro_bic < reel_simplifie
        df.loc[indexes, "impot"] = np.where(loyer_condition,
                                             np.where(regime_condition, micro_bic, reel_simplifie),
                                             reel_simplifie)
        df.loc[indexes, "regime"] = np.where(loyer_condition,
                                             np.where(regime_condition, 'micro_bic', 'reel_simplifie'),
                                             'reel_simplifie')
        df["regime"].fillna(method="bfill", inplace=True)
        df.loc[indexes, "reel_simplifie"] = reel_simplifie.values
        df.loc[indexes, "micro_bic"] = micro_bic.values
        df = df[["amortissement", "reel_simplifie", "micro_bic", "impot", "regime"]]
        df.fillna(0, inplace=True)
        return df

class InvestissementLocatif:
    """
    Une classe modélisant un investissement locatif en LMNP.
    
    Les méthodes calculent et affichent les différent taux de rendement de l'investissement
    ainsi que les cashflows annuels et les différents coût d'apport pour acquérir le bien.
    
    Attributs
    ----------
    list_depense : list
        List regroupant les noms des charges/depenses.
    list_profit : list
        List regroupant les noms des revenus.
    list_apport :
        List regroupant les noms des differents couts fixe d'acquisition (avant début du prêt).
    r_plus_value: float
        Taux d'imposition sur la plus value à la revente.
    r_prelevement_sociaux : float
        Taux des prélèvement sociaux sur la plus value à la revente.
    r_bien : float
        Taux de rendement pour l'appréciation du prix du bien.
    loyer: float
        Loyer (mensuel).
    frais_agence_vente : float
        Taux de frais d'agence par rapport aux prix du bien à la revente.
    notaire: Notaire
        Objet de la classe Notaire.
    fiscalite : Fiscalite
        Objet de la classe Fiscalite.
    df : DataFrame
        DataFrame regroupant tous les flux entrant et sortant de l'investissement.
    """
    def __init__(self,
                 prix_achat: float,
                 loyer: float,
                 surface: float,
                 tp_bien: str,
                 r_bien: float,
                 r_banque: float,
                 r_assurance: float,
                 r_impot: float,
                 r_plus_value: float,
                 r_prelevement_sociaux: float,
                 start_date_achat: Optional[str],
                 start_date_loyer: Optional[str],
                 T: Optional[float],
                 T_differe: Optional[float],
                 travaux: Optional[float],
                 pret_travaux: Optional[bool],
                 meubles: Optional[float],
                 pret_meubles: Optional[bool],
                 taux_apport: Optional[float],
                 r_loyer: Optional[float],
                 frais_notaire: Optional[float],
                 frais_prorata: Optional[float],
                 frais_signature: Optional[float],
                 frais_agence_achat: Optional[float],
                 frais_agence_vente: Optional[float],
                 frais_garantis: Optional[float],
                 frais_dossier: Optional[float],
                 frais_assurance_emprunt: Optional[float],
                 frais_comptable: Optional[float],
                 frais_gestion: Optional[float],
                 frais_service: Optional[float],
                 assurance_GLI: Optional[float],
                 assurance_habitation: Optional[float],
                 copropriete: Optional[float],
                 amortissement: Optional[Dict]
                ) -> None:
        """
        Initialisation.
        
        Parametres obligatoires
        -----------------------
        prix_achat : float
            Prix du bien à l'achat.
        loyer : float
            Loyer (mensuel).
        surface : float
            Surface du bien (m2).
        tp_bien : str
            Type du bien. Entrée valide: "ancien" ou "neuf".
            Un logement est considéré 'neuf' si il a été crée il y a moins de 5ans et n'a jamais été habité ni loué.
        r_bien : float
            Taux de rendement pour l'appréciation du prix du bien.
        r_banque : float
            Taux d'intéret bancaire.
        r_assurance : float
            Taux d'assurance TAEA (annuel).
            Simulateur: https://www.maaf.fr/pretimmobilier/simulation-assurance-pret-immobilier/tarif
        r_impot : float
            Taux d'imposition.
            r = r1 + r2
            avec: - r1: tranche d'imposition de l'investisseur.
                  - r2: prélèvements sociaux si résidents fiscaux francais OU prélèvements solidarité sinon.
        r_plus_value : float
            Taux d'imposition sur la plus value à la revente.
        r_prelevement_sociaux : float
            Taux des prélèvement sociaux sur la plus value à la revente.
            
        Parametres optionnels
        ---------------------
        start_date_achat : str
            Date de commencement du prêt.
        start_date_loyer : str
            Date du premier loyer percu.
        T : float
            Durée total du prêt (en année).
        T_differe : float
            Durée du prêt en différé (en année).
            Durant cette période, l'emprunteur paye les intérets sans rembourser le capital.
            ('T' - 'T_differe') représente le nombre d'années en prêt classique.
        travaux: float
            Prix des travaux.
        pret_travaux: bool
            True si les travaux sont financés par le prêt. Faux sinon.
        meubles: float
            Prix des travaux.
        pret_meubles: bool
            True si l'ammeublement est financé par le prêt. Faux sinon.
        taux_apport : float
            Taux d'apport.
        r_loyer : float
            Taux annuel d'accroisement des loyers.
        frais_notaire : float
            Frais du notaire.
        frais_prorata : float
            Frais payés à l'avance par le vendeur à rembourser.
        frais_signature : float
            Frais de signature du notaire.
        frais_agence_achat : float
            Frais d'agence à l'achat.
        frais_agence_vente : float
            Taux de frais d'agence par rapport aux prix du bien à la revente.
        frais_garantis : float
            Frais de la garantis d'emprunt.
        frais_dossier : float
            Frais de dossier bancaire.
        frais_assurance_emprunt : float
            Frais de l'assurance emprunt.
        frais_comptable : float
            Frais comptable (annuel).
        frais_gestion : float
            Taux retranché aux loyers pour payer l'agence de gestion.
        frais_service: float
            Frais (mensuel) de service tel que wifi, electricité, gaz.
            En général, ces charges sont payés par le locataire.
        assurance_GLI : float
            Assurance garantis de loyer impayés. Taux retranché aux loyers pour payer l'assurance.
            En général cette assurance est nulle (soit payé par le locataire, soit le locataire possède un garant).
        assurance_habitation : float
            Frais (annuel) d'assurance pour le bien contre les sinistres.
        copropriete: float
            Frais de copropriété (trimestriel) pour les charges courantes de l'immeuble. Des appels de roulement sont
            possible pour réapprovisionner le fond de trésorerie si utilisé. Un fond de réserve pour la réalisation de futures
            gros travaux peut également être demander.
        amortissement : Dict.
            Charges amortissables: "structure", "IGT", "agencement", "facade", "travaux", "meubles".
            key: str, nom de la charge.
            value: dict, ('montant': montant total de la charge, 'duree': nombre d'années amortissable).
        """
        if tp_bien not in ["ancien", "neuf"]:
            return ValueError("'tp_bien' = 'ancien' ou 'neuf'")
        if start_date_achat is None:
            start_date_achat = (pd.Timestamp.today() + DateOffset(months=2)).strftime("%Y-%m-%d")
        if start_date_loyer is None:
            start_date_loyer = (pd.Timestamp.today() + DateOffset(months=4)).strftime("%Y-%m-%d")
        if T is None:
            T = 20
        if T_differe is None:
            T_differe = 0
        if travaux is None:
            travaux = 0
        if pret_travaux is None:
            pret_travaux = False
        if meubles is None:
            meubles = 0
        if pret_meubles is None:
            pret_meubles = False
        if taux_apport is None:
            taux_apport = 0
        emprunt = prix_achat + travaux * (1 if pret_travaux else 0) + meubles * (1 if pret_meubles else 0)
        emprunt *= (1 - taux_apport)
        if r_loyer is None:
            r_loyer = 0
        if copropriete is None:
            copropriete = 35 * surface / 4
        if frais_notaire is None:
            frais_notaire = prix_achat * (7.5 / 100 * (tp_bien == "ancien") + 2.5 / 100 * (tp_bien == "neuf"))
        if frais_prorata is None: # ADD TAXE FONCIERE
            date_vente = date_vente = pd.to_datetime(start_date_achat)
            prorata = (pd.date_range(start=date_vente, periods=1, freq="QS")[0] - date_vente).days / (30.5 * 3)
            frais_prorata = copropriete * prorata
        if frais_signature is None:
            frais_signature = 10
        if frais_agence_achat is None:
            frais_agence_achat = 0
        if frais_agence_vente is None:
            frais_agence_vente = 7 / 100
        if frais_garantis is None:
            commission = min(max(180, emprunt * 0.5 / 100), 1020)
            FMG = 1 / 100 * emprunt
            frais_garantis = commission + FMG # https://www.creditlogement.fr/particuliers/simulateur-frais-de-garantie/
        if frais_dossier is None:
            frais_dossier = min(max(500, 0.75 / 100 * emprunt), 2000)
        if frais_assurance_emprunt is None:
            frais_assurance_emprunt = 50
        if frais_comptable is None:
            frais_comptable = 400
        if frais_gestion is None:
            frais_gestion = 5 / 100
        if frais_service is None:
            frais_service = 0
        if assurance_GLI is None:
            assurance_GLI = 0
        if assurance_habitation is None:
            assurance_habitation = surface * 8
        
        notaire = Notaire(prix_achat=prix_achat,
                          travaux=travaux,
                          meubles=meubles,
                          frais_notaire=frais_notaire,
                          frais_agence_achat=frais_agence_achat,
                          frais_prorata=frais_prorata,
                          frais_signature=frais_signature)
        
        N = emprunt / (1 - taux_apport)
        financement = Financement(start_date_achat, N=N, taux_apport=taux_apport, T=T, T_differe=T_differe)
        df_interet = financement.calcul_interet(r=r_banque,
                                                frais_garantis=frais_garantis,
                                                frais_dossier=frais_dossier)
        df_assurance = financement.calcul_assurance(r=r_assurance,
                                                    frais_assurance_emprunt=frais_assurance_emprunt,
                                                    capital_restant=df_interet["capital_restant"])
        df_financement = df_interet.merge(df_assurance, left_index=True, right_index=True)
        
        cout_travaux = 0 if pret_travaux else travaux
        cout_meubles = 0 if pret_meubles else meubles
        bien_immobilier = BienImmobilier(start_date_achat=start_date_achat,
                                         start_date_loyer=start_date_loyer,
                                         loyer=loyer,
                                         cout_travaux=cout_travaux,
                                         cout_meubles=cout_meubles)
        df_bien_immobilier = bien_immobilier.calcul_frais(r_loyer=r_loyer,
                                                          frais_gestion=frais_gestion,
                                                          assurance_GLI=assurance_GLI,
                                                          assurance_habitation=assurance_habitation,
                                                          copropriete=copropriete,
                                                          frais_service=frais_service)
        
        fiscalite = Fiscalite(r=r_impot, frais_comptable=frais_comptable, notaire=notaire, amortissement=amortissement)
        df = df_financement.merge(df_bien_immobilier, left_index=True, right_index=True)
        df_fiscalite = fiscalite.calcul_impot(df=df)
        
        df = pd.concat([df_financement, df_bien_immobilier, df_fiscalite], axis=1)
        start = df_bien_immobilier[df_bien_immobilier["loyer"] > 0].index[0]
        indexes = pd.date_range(start=start, end=df.index[-1], freq=DateOffset(years=1))
        df.loc[indexes, "frais_comptable"] = frais_comptable
        df.loc[df.index[0], "frais_notaire"] = frais_notaire
        df.loc[df.index[0], "frais_prorata"] = frais_prorata
        df.loc[df.index[0], "frais_signature"] = frais_signature
        df.loc[df.index[0], "frais_agence_achat"] = frais_agence_achat
        df.fillna(0, inplace=True)
        
        self.list_depense = ["interet", "capital", "prime_assurance", "frais_gestion", "assurance_GLI",
                            "assurance_habitation", "copropriete", "frais_service", "frais_sinistre", "impot",
                            "frais_comptable", "cout_travaux", "frais_signature", "frais_prorata", "apport_pret",
                            "cout_meubles", "frais_garantis", "frais_notaire", "frais_assurance_emprunt", "frais_dossier",
                            "frais_agence_achat"]
        self.list_profit = ["loyer"]
        self.list_apport = ["frais_notaire", "frais_prorata", "frais_signature", "frais_agence", "apport_pret",
                            "frais_garantis", "frais_dossier", "frais_assurance_emprunt", "cout_travaux", "cout_meubles"]
        self.r_banque = r_banque
        self.r_plus_value = r_plus_value
        self.r_prelevement_sociaux = r_prelevement_sociaux
        self.r_bien = r_bien
        self.loyer = loyer
        self.frais_agence_vente = frais_agence_vente
        self.restitution_FMG = 74 / 100 * (1 / 100 * emprunt)
        self.notaire = notaire
        self.fiscalite = fiscalite
        self.df = df
        
    def apport(self) -> Dict:
        """
        Calcul les différent apports pour acquérir le bien:
            - apport_notaire = frais_notaire + frais_prorata + frais_signature
            - apport_agence = frais_agence_achat
            - apport_banque = apport du prêt + frais_garantis + frais_dossier + frais_assurance_emprunt
            - apport_bien = cout_travaux (si non financé par le prêt) + cout_meubles (si non financé par le prêt)
        """
        apport_notaire = self.df[["frais_notaire", "frais_prorata", "frais_signature"]].sum().sum()
        apport_agence = self.df[["frais_agence_achat"]].sum().sum()
        apport_banque = self.df[["apport_pret", "frais_garantis", "frais_dossier", "frais_assurance_emprunt"]].sum().sum()
        apport_bien = self.df[["cout_travaux", "cout_meubles"]].sum().sum()
        apport_total = apport_notaire + apport_agence + apport_banque + apport_bien
        apport = {
            "notaire": apport_notaire,
            "agence": apport_agence,
            "banque": apport_banque,
            "bien": apport_bien,
            "total": apport_total
        }
        return apport
        
    def cashflow_annuel(self) -> pd.Series:
        """
        Calcul les cashflows annuel (profit - depense).
        """
        def CF(x: pd.DataFrame) -> pd.Series:
            """
            Calcul les cashflows pour une année.
            
            Parametres
            ----------
            x : Groupby Dataframe
                Groupby annuel.
            """
            depense = x[self.list_depense].sum().sum()
            profit = x[self.list_profit].sum().sum()
            cashflow = profit - depense
            return cashflow
        
        cashflow = self.df.groupby(self.df.index.year).apply(CF)
        return cashflow
    
    def calcul_pourcentage_financement(self, date_vente: str) -> float:
        """
        Calcul le pourcentage du bien à payer par l'acquéreur à la date de revente.
        
        - Si les dépenses dépassent les loyers, le taux est positif:
            Le pourcentage représente la part du bien que l'acquéreur a dû payer de sa poche.
        
        - Si les loyers dépassent les dépenses, le taux devient négatif:
            L'acquéreur n'a jamais du sortir d'argent de sa poche pour payer le bien et recoit en plus une rémunération.
        
        Parametres
        ----------
        date_vente : str
            Date de revente du bien.
        """
        depense = self.df.loc[self.df.index <= date_vente, self.list_depense].sum().sum()
        profit = self.df.loc[self.df.index <= date_vente, self.list_profit].sum().sum()
        revente_bien = self.revente_bien(date_vente)[1]
        pourcentage_paye = (depense - profit) / revente_bien
        return pourcentage_paye
    
    def rendement(self, inclus_frais_acquisition: bool, inclus_financement: bool, inclus_imposition: bool, inclus_depense: bool, ) -> float:
        """
        Calcul les différent taux de rendements de la forme:
            r = (12 * AVG(loyer) - AVG(depense_annuel)) / cout_acquisition
            avec depense_annuel et cout_acquisition dépendant des paramètres.
            
        Parametres
        ----------
        inclus_frais_acquisition : bool
            Si False, cout_acquisition = prix d'achat + travaux + meubles
            Si True, on ajout en plus: frais_notaire, frais_signature, frais_agence_achat.
        inclus_financement:  bool
            Si False, les depenses annuel et cout d'acquisition n'incluent pas les financements du prêt.
            Si True, on ajoute aux dépenses annuels: interet et prime d'assurance.
                     on ajoute au cout d'acquisition: frais de garantis, frais d'assurance emprunt, frais de dossier bancaire.
        inclus_imposition : bool
            Si False, les depenses annuel n'incluent pas l'imposition et les frais de comptable.
            Si True, les depenses annuel incluent l'imposition et les frais de comptable.
        is_depense : bool
            Si False, depense_annuel = 0
            Si True, depense_annuel dépend de 'inclus_financement' et 'inclus_imposition'.
        """
        start = self.df.index[0].year + 1
        end = self.df[self.df["capital_restant"]==0].index[0].year - 1
        loyer_annuel = self.df.groupby(self.df.index.year)[self.list_profit].sum().sum(axis=1)
        loyer_annuel = loyer_annuel.loc[start:end].mean()
        list_depense = ["assurance_GLI", "copropriete", "assurance_habitation", "frais_service", "frais_gestion", "frais_sinistre"]
        cout_acquisition = self.notaire.prix_achat + self.notaire.travaux + self.notaire.meubles
        if inclus_frais_acquisition:
            cout_acquisition += self.df[["frais_notaire", "frais_signature", "frais_agence_achat"]].sum().sum()
        if inclus_financement:
            cout_acquisition += self.df[["frais_garantis", "frais_assurance_emprunt", "frais_dossier"]].sum().sum()
            list_depense += ["interet", "prime_assurance"]
        if inclus_imposition:
            list_depense += ["impot", "frais_comptable"]
        ### ADD TAXE FONCIERE TO list_depense ###
        if inclus_depense:
            depense_annuelle = self.df.groupby(self.df.index.year)[list_depense].sum().sum(axis=1)
            depense_annuelle = depense_annuelle.loc[start:end].mean()
        else:
            depense_annuelle = 0
        r = (loyer_annuel - depense_annuelle) / cout_acquisition
        return r
    
    def calcul_TIR(self, date_vente: str) -> float:
        """
        Calcul le taux de rentabilité interne (TIR) jusqu'à 'date_vente'.
        C'est le taux r tel que la somme des discounted cashflow au taux r sur [0, 'date_vente'] valent 0:
        SUM(t, DISC_r(Cashflow_t)) = 0 <==> SUM('date_vente'>t>0, DISC_r(Cashflow_t)) + DISC_r(Cashflow_0) = 0
                                       <==> SUM('date_vente'>t>0, DISC_r(Cashflow_t)) + Cashflow_0 = 0
                                       <==> SUM('date_vente'>t>0, DISC_r(Cashflow_t)) = -Cashflow_0
            avec Cashflow_0 < 0 en générale (recette < dépense à t=0)
            
        Ainsi, en investissant |Cashflow_0| au taux r, on réplique tous les cashflows futures sur ]0, 'date_vente'].
            
        Parametres
        ----------
        date_vente : str
            Date de revente du bien.
        """
        def f(r: float, df: pd.DataFrame) -> float:
            """
            Function à optimiser pour trouver r tel que f(r)=0.
            
            Parametres
            ----------
            r : float
                taux de rentabilité interne à optimiser.
            df : DataFrame.
                copy de 'self.df'.
            """
            profit = df[self.list_profit+["prix_revente"]].sum(axis=1)
            depense = df[self.list_depense].sum(axis=1)
            df["cashflow"] = profit - depense
            df["discount_rate"] = 1 / (1 + r)**(df["time_to_maturity"])
            df["present_value"] = df["cashflow"] * df["discount_rate"]
            VAN = df["present_value"].sum()
            return VAN

        df = self.df.copy()
        df["prix_revente"] = 0
        df = df.loc[df.index <= date_vente]
        df["time_to_maturity"] = [(d - df.index[0]).days / 365 for d in df.index]
        valeur_bien_vente_brut, valeur_bien_vente_net = self.revente_bien(date_vente=date_vente)
        df.loc[df.index[-1], "prix_revente"] = valeur_bien_vente_net
        
        try:
            r_annuel = root_scalar(f, x0=2/100, bracket=[-0.2,10], args=(df,), xtol=1e-4, method='bisect').root
        except ValueError as e:
            r_annuel = np.nan
        return r_annuel
    
    def revente_bien(self, date_vente: str) -> Tuple[float, float]:
        """
        Calcul la valeur de revente brut et net (en retranchant les frais d'agence, d'imposition, de capital restant dû,
        et de pénalité d'interets) du bien.

        Parametres
        ----------
        date_vente : str
            Date de revente du bien.
        """
        valeur_bien_achat = self.notaire.prix_achat + self.notaire.frais_notaire + self.notaire.frais_agence_achat
        duree_detention = (self.df.loc[self.df.index <= date_vente].index[-1] - self.df.index[0]).days / 365
        valeur_bien_vente_brut = self.notaire.prix_achat * (1 + self.r_bien)**(duree_detention)
        frais_agence_vente = valeur_bien_vente_brut * self.frais_agence_vente
        
        v = np.cumsum([0] * 5 + [6/100] * 16 + [4/100])
        k = range(1, 23)
        d = dict(zip(k, v))
        D = min(int(duree_detention), max(k))
        r_plus_value = (1 - d.get(D, 0)) * self.r_plus_value
        
        v = np.cumsum([0] * 5 + [1.65/100] * 16 + [1.6/100] + [9/100] * 8)
        k = range(1, len(v)+1)
        d = dict(zip(k, v))
        D = min(int(duree_detention), max(k))
        r_prelevement_sociaux = (1 - d.get(D, 0)) * self.r_prelevement_sociaux
        
        impot_plus_value = (valeur_bien_vente_brut - frais_agence_vente - valeur_bien_achat) * (r_plus_value + r_prelevement_sociaux)
        impot_plus_value = max(impot_plus_value, 0)
        capital_restant = self.df.loc[self.df.index <= date_vente, "capital_restant"].iloc[-1]
        indemnite = min(0.03, self.r_banque / 2) * capital_restant
        restitution = self.restitution_FMG
        valeur_bien_vente_net = (valeur_bien_vente_brut + restitution - frais_agence_vente - impot_plus_value
                                 - capital_restant - indemnite)
        return valeur_bien_vente_brut, valeur_bien_vente_net

@dataclass
class MC():
    """
    Classe ajoutant des coûts aléatoire dû aux locations vacantes ou aux sinistres survenus.
    Les métriques moyennes sont calculés par méthode de Monte Carlo.
    
    Attributs
    ----------
    investissement : InvestissementLocatif
        Objet de la classe InvestissementLocatif.
    r_innocupation : float
        Frequence d'innocupation du bien.
    r_sinistre : float
        Frequence de sinistres survenus.
    frais_innocupation : float
        Frais de gestion additionnels lorsque le logement est vacant (recherche du nouvel locataire).
    frais_sinistre : float
        Frais moyen d'un sinistre.
    K : int
        Nombre d'échantillons Monte Carlo.
    chemins : List[InvestissementLocatif]
        Chemins Monte Carlo.
        
    Notes
    -----
    L'attribut 'chemins' est calculé automatiquement lors de l'initialisation par la méthode 'simulation'.
    """
    
    def __init__(self, investissement: InvestissementLocatif, r_innocupation: float, r_sinistre: float, frais_innocupation: float, frais_sinistre: float, K: int) -> None:
        """
        Initialisation.
        
        Parametres obligatoires
        -----------------------
        investissement : InvestissementLocatif
            Objet de la classe InvestissementLocatif.
        
        Parametres optionnels
        ---------------------
        r_innocupation : float
            Frequence d'innocupation du bien.
        r_sinistre : float
            Frequence de sinistres survenus.
        frais_innocupation : float
            Frais de gestion additionnels lorsque le logement est vacant (recherche du nouvel locataire).
        frais_sinistre : float
            Frais moyen d'un sinistre.
        K : int
            Nombre d'échantillons Monte Carlo.
        """
        if r_innocupation is None:
            r_innocupation = 1 / 18
        if r_sinistre is None:
            r_sinistre = 1 / 12
        if frais_innocupation is None:
            frais_innocupation = 60 / 100 * investissement.loyer
        if frais_sinistre is None:
            frais_sinistre = 50 / 100 * investissement.loyer
        if K is None:
            K = 30
        self.investissement = investissement
        self.r_innocupation = r_innocupation
        self.r_sinistre = r_sinistre
        self.frais_innocupation = frais_innocupation
        self.frais_sinistre = frais_sinistre
        self.K = K
        self.simulation()
        
    def simulation(self) -> None:
        """
        Simule et stock les 'self.K' chemins Monte Carlo.
        """
        def f() -> InvestissementLocatif:
            """
            Simule un chemin Monte Carlo en ajoutant les vacance locatives et les frais d'innocupation et de sinistre.
            
            L'imposition au régime réél simplifié ou micro bic est recalculé en considérant les frais additionnels.
            Cependant, le régime optimal (imposition minimale) est choisi avant l'ajout des frais additionnels car aléatoires.
            """
            investissement = copy.deepcopy(self.investissement)
            n = len(investissement.df)
            is_vacant = np.random.random(size=n) < self.r_innocupation
            is_sinistre = np.random.random(size=n) < self.r_sinistre
            investissement.df["loyer"] *= (1 - is_vacant)
            investissement.df["frais_gestion"] += is_vacant * self.frais_innocupation
            investissement.df["frais_sinistre"] += is_sinistre * self.frais_sinistre
            df_fiscalite = investissement.fiscalite.calcul_impot(investissement.df)
            investissement.df["micro_bic"] = df_fiscalite["micro_bic"]
            investissement.df["reel_simplifie"] = df_fiscalite["reel_simplifie"]
            investissement.df["impot"] = np.where(investissement.df["regime"] == "micro_bic",
                                                  df_fiscalite["micro_bic"],
                                                  df_fiscalite["reel_simplifie"])
            return investissement
        self.chemins = Parallel(n_jobs=-1)(delayed(f)() for _ in range(self.K))
        
    def cashflow_annuel(self) -> pd.Series:
        """
        Calcul les cashflows annuel (profit - depense) moyen avec et sans les coûts aléatoires additionnels.
        """
        cashflow = self.investissement.cashflow_annuel()
        cashflows_MC = sum([investissement.cashflow_annuel() for investissement in self.chemins]) / self.K
        df = pd.DataFrame({"CF_sans_cout_aleatoire": cashflow, "CF_avec_cout_aleatoire": cashflows_MC})
        return df

    def afficher_metriques(self, inclus_frais_acquisition: bool) -> None:
        """
        Calcul et affiche les métriques d'investissement moyen avec et sans les coûts aléatoires additionnels:
            - apport initial
            - valorisation du bien initial et final
            - pourcentage financement
            - rendement brut
            - cap rate
            - rendement net
            - rendement net net
            - TIR
        Le rendement TIR est affiché graphiquement: (t, TIR(t))
            avec t = durée de détention du bien (maturité).
        
        Parametres
        ----------
        inclus_frais_acquisition : bool
            Inclus ou non les frais de notaire, prorata et de signature pour calculer les rendements.
        """
        dates = pd.date_range(start=self.investissement.df.index[0],
                              end=self.investissement.df.index[-1],
                              freq=DateOffset(years=1)).strftime('%Y-%m-%d')
        T = len(self.investissement.df[self.investissement.df["interet"] > 0]) // 12
        pourcentage_paye = self.investissement.calcul_pourcentage_financement(date_vente=dates[T])
        pourcentage_paye_MC = np.mean([investissement.calcul_pourcentage_financement(date_vente=dates[T])
                            for investissement in self.chemins])
        rendement_brut = self.investissement.rendement(inclus_frais_acquisition=inclus_frais_acquisition,
                                                    inclus_financement=None, inclus_imposition=None, inclus_depense=False)
        rendement_brut_MC = np.mean([investissement.rendement(inclus_frais_acquisition=inclus_frais_acquisition,
                                                     inclus_financement=None, inclus_imposition=None, inclus_depense=False)
                            for investissement in self.chemins])
        cap_rate = self.investissement.rendement(inclus_frais_acquisition=inclus_frais_acquisition,
                                                     inclus_financement=False, inclus_imposition=False, inclus_depense=True)
        cap_rate_MC = np.mean([investissement.rendement(inclus_frais_acquisition=inclus_frais_acquisition,
                                                     inclus_financement=False, inclus_imposition=False, inclus_depense=True)
                            for investissement in self.chemins])
        rendement_net = self.investissement.rendement(inclus_frais_acquisition=inclus_frais_acquisition,
                                                     inclus_financement=True, inclus_imposition=False, inclus_depense=True)
        rendement_net_MC = np.mean([investissement.rendement(inclus_frais_acquisition=inclus_frais_acquisition,
                                                     inclus_financement=True, inclus_imposition=False, inclus_depense=True)
                            for investissement in self.chemins])
        rendement_net_net = self.investissement.rendement(inclus_frais_acquisition=inclus_frais_acquisition,
                                                     inclus_financement=True, inclus_imposition=True, inclus_depense=True)
        rendement_net_net_MC = np.mean([investissement.rendement(inclus_frais_acquisition=inclus_frais_acquisition,
                                                     inclus_financement=True, inclus_imposition=True, inclus_depense=True)
                            for investissement in self.chemins])
        TIRs = list(map(lambda x: self.investissement.calcul_TIR(date_vente=x), dates))
        TIRs_MC = np.mean([np.array([investissement.calcul_TIR(date_vente=date_vente) for date_vente in dates])
                        for investissement in self.chemins], axis=0)
        apport = self.investissement.apport()
        valeur_bien_vente = self.investissement.revente_bien(date_vente=dates[T])

        plt.figure(figsize=(12, 5))
        x = range(len(TIRs_MC))
        y = np.array([round(100 * r, 2) for r in TIRs])
        y_MC = np.array([round(100 * r, 2) for r in TIRs_MC])
        nan_indices = np.isnan(y) | np.isnan(y_MC)
        y[nan_indices], y_MC[nan_indices] = np.nan, np.nan
        plt.plot(x, y, color="blue", label="sans coûts aléatoires")
        plt.plot(x, y_MC, color="red", label="avec coûts aléatoires")
        plt.axvline(x=T, color='black', linestyle='--')
        plt.text(T+0.5, int(min(y[~np.isnan(y)])), f'r={y_MC[T]}% (r={y[T]}%)', color='black', ha='center', rotation=90)
        plt.title("Taux de rentabilité interne")
        plt.xlabel("durée détention du bien (années)")
        plt.ylabel("rendement (%)")
        plt.legend()
        plt.show()
        print(f"aport initial: {apport}")
        print("\n")
        print(f"valorisation bien (T=0): {round(self.investissement.notaire.prix_achat)}")
        print(f"valorisation bien brute (T={T}): {round(valeur_bien_vente[0])}")
        print(f"valorisation bien net (T={T}): {round(valeur_bien_vente[1])}")
        print("\n")
        print(f"pourcentage du bien payé (T={T}): {round(100 * pourcentage_paye_MC, 2)}% ({round(100 * pourcentage_paye, 2)}% sans coûts aléatoires)")
        print("[<5%]: excellent | [5%,10%]: très bon | [10%,20%]: OK | [>20%]: faible")
        print("\n")
        print(f"rendement brut: {round(100 * rendement_brut_MC, 2)}% ({round(100 * rendement_brut, 2)}% sans coûts aléatoires)")
        print("[<5%]: faible | [5%,7%]: OK | [7%,10%]: très bon | [>10%]: excellent")
        print("\n")
        print(f"taux capitalisation (+frais): {round(100 * cap_rate_MC, 2)}% ({round(100 * cap_rate, 2)}% sans coûts aléatoires)")
        print("[<5%]: faible | [5%,7%]: OK | [7%,10%]: très bon | [>10%]: excellent")
        print("\n")
        print(f"rendement_net (+financement): {round(100 * rendement_net_MC, 2)}% ({round(100 * rendement_net, 2)}% sans coûts aléatoires)")
        print("[<4%]: faible | [4%,6%]: OK | [6%,8%]: très bon | [>8%]: excellent")
        print("\n")
        print(f"rendement_net_net (+impot): {round(100 * rendement_net_net_MC, 2)}% ({round(100 * rendement_net_net, 2)}% sans coûts aléatoires)")
        print("[<3%]: faible | [3%,5%]: OK | [5%,7%]: très bon | [>7%]: excellent")
