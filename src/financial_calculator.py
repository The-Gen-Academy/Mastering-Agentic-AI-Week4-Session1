"""
Financial Calculation Engine for CardWise AI
Calculates point earnings, statement credit utilization, and Net Annual Value
for Chase Sapphire Reserve and American Express Platinum Card.
"""

from typing import Dict, Any, Optional


class CardWiseFinancialCalculator:
    """
    Deterministic financial calculator that evaluates spend profiles
    against rewards structures and annual fees for both cards.
    """

    # Annual Fees
    CSR_ANNUAL_FEE = 795.0
    AMEX_PLAT_ANNUAL_FEE = 895.0

    # Default Point Valuations (in dollars per point)
    DEFAULT_CSR_POINT_VALUATION = 0.015  # 1.5 cents per point (standard Sapphire travel value)
    DEFAULT_AMEX_POINT_VALUATION = 0.015  # 1.5 cents per point (average transfer partner value)

    def calculate_card_value(
        self,
        spend_profile: Dict[str, float],
        credits_utilized: Optional[Dict[str, float]] = None,
        csr_point_value: float = DEFAULT_CSR_POINT_VALUATION,
        amex_point_value: float = DEFAULT_AMEX_POINT_VALUATION
    ) -> Dict[str, Any]:
        """
        Calculates detailed financial breakdown and Net Annual Value for both cards.

        spend_profile keys (annual $ spend):
          - 'flights_direct': $ spent on flights directly with airlines
          - 'flights_portal': $ spent on flights via Chase Travel / Amex Travel
          - 'hotels_direct': $ spent on hotels directly
          - 'hotels_portal': $ spent on hotels via portal (Chase Travel / Amex Travel)
          - 'dining': $ spent on dining/restaurants
          - 'lyft': $ spent on Lyft
          - 'other_spend': $ spent on all other general purchases
        """
        spend = {
            "flights_direct": spend_profile.get("flights_direct", 0.0),
            "flights_portal": spend_profile.get("flights_portal", 0.0),
            "hotels_direct": spend_profile.get("hotels_direct", 0.0),
            "hotels_portal": spend_profile.get("hotels_portal", 0.0),
            "dining": spend_profile.get("dining", 0.0),
            "lyft": spend_profile.get("lyft", 0.0),
            "other_spend": spend_profile.get("other_spend", 0.0)
        }

        credits = credits_utilized or {}

        # ----------------------------------------------------
        # 1. CHASE SAPPHIRE RESERVE (CSR) CALCULATIONS
        # ----------------------------------------------------
        # Calculate $300 Annual Travel Credit offset first (travel spend reimbursed doesn't earn points)
        total_travel_spend = (
            spend["flights_direct"] + spend["flights_portal"] +
            spend["hotels_direct"] + spend["hotels_portal"] +
            spend["lyft"]
        )
        csr_travel_credit_used = min(300.0, total_travel_spend)

        # Calculate CSR Points
        csr_points_chase_travel = (spend["flights_portal"] + spend["hotels_portal"]) * 8.0
        csr_points_direct_flights = spend["flights_direct"] * 4.0
        csr_points_direct_hotels = spend["hotels_direct"] * 4.0
        csr_points_dining = spend["dining"] * 3.0
        csr_points_lyft = spend["lyft"] * 5.0
        csr_points_other = spend["other_spend"] * 1.0

        total_csr_points = (
            csr_points_chase_travel + csr_points_direct_flights +
            csr_points_direct_hotels + csr_points_dining +
            csr_points_lyft + csr_points_other
        )

        csr_points_cash_value = total_csr_points * csr_point_value

        # Calculate CSR Statement Credits
        csr_credits_breakdown = {
            "Annual Travel Credit": csr_travel_credit_used,
            "Chase Travel Hotel Credit (2026)": min(250.0, credits.get("csr_hotel_2026", 250.0 if spend["hotels_portal"] >= 250 else 0.0)),
            "The Edit Hotel Credit": min(500.0, credits.get("csr_edit_hotel", 0.0)),
            "OpenTable Dining Credit": min(300.0, credits.get("csr_opentable", 0.0)),
            "DoorDash Membership & Promos": min(300.0, credits.get("csr_doordash", 150.0)),
            "Apple Subscriptions": min(288.0, credits.get("csr_apple", 0.0)),
            "StubHub Credit": min(300.0, credits.get("csr_stubhub", 0.0)),
            "Lyft Credit": min(120.0, credits.get("csr_lyft", 120.0 if spend["lyft"] > 0 else 0.0)),
            "Peloton Credit": min(120.0, credits.get("csr_peloton", 0.0))
        }

        # Check $75k high spend threshold
        total_annual_spend = sum(spend.values())
        if total_annual_spend >= 75000.0:
            csr_credits_breakdown["High Spend Perks ($750 Credits)"] = 750.0

        total_csr_credits_val = sum(csr_credits_breakdown.values())
        csr_net_annual_value = csr_points_cash_value + total_csr_credits_val - self.CSR_ANNUAL_FEE

        # ----------------------------------------------------
        # 2. AMEX PLATINUM CARD CALCULATIONS
        # ----------------------------------------------------
        # Amex 5X Flights (direct or Amex travel portal) capped at $500k/yr
        total_flight_spend = spend["flights_direct"] + spend["flights_portal"]
        flight_5x_eligible = min(500000.0, total_flight_spend)
        flight_1x_remainder = max(0.0, total_flight_spend - 500000.0)

        amex_points_flights = (flight_5x_eligible * 5.0) + (flight_1x_remainder * 1.0)
        amex_points_hotels_portal = spend["hotels_portal"] * 5.0
        amex_points_hotels_direct = spend["hotels_direct"] * 1.0
        amex_points_dining = spend["dining"] * 1.0
        amex_points_lyft = spend["lyft"] * 1.0
        amex_points_other = spend["other_spend"] * 1.0

        total_amex_points = (
            amex_points_flights + amex_points_hotels_portal +
            amex_points_hotels_direct + amex_points_dining +
            amex_points_lyft + amex_points_other
        )

        amex_points_cash_value = total_amex_points * amex_point_value

        # Calculate Amex Credits
        amex_credits_breakdown = {
            "Hotel Credit (FHR/THC)": min(600.0, credits.get("amex_hotel_credit", 600.0 if spend["hotels_portal"] >= 600 else 0.0)),
            "Airline Fee Credit": min(200.0, credits.get("amex_airline_fee", 200.0 if total_flight_spend > 0 else 0.0)),
            "Uber Cash & Uber One": min(320.0, credits.get("amex_uber", 200.0)),
            "Resy Dining Credit": min(400.0, credits.get("amex_resy", 200.0 if spend["dining"] >= 200 else 0.0)),
            "Digital Entertainment": min(300.0, credits.get("amex_digital_ent", 150.0)),
            "lululemon Credit": min(300.0, credits.get("amex_lululemon", 0.0)),
            "Walmart+ Credit": min(155.0, credits.get("amex_walmart", 0.0)),
            "CLEAR+ Credit": min(219.0, credits.get("amex_clear", 219.0 if total_flight_spend > 0 else 0.0)),
            "Oura Ring Credit": min(200.0, credits.get("amex_oura", 0.0)),
            "Equinox Credit": min(300.0, credits.get("amex_equinox", 0.0))
        }

        total_amex_credits_val = sum(amex_credits_breakdown.values())
        amex_net_annual_value = amex_points_cash_value + total_amex_credits_val - self.AMEX_PLAT_ANNUAL_FEE

        # Recommendation logic
        if csr_net_annual_value > amex_net_annual_value:
            recommendation = "Chase Sapphire Reserve"
            net_difference = csr_net_annual_value - amex_net_annual_value
            reason = f"Yields ${net_difference:.2f} higher Net Annual Value based on your spend profile and credit usage."
        elif amex_net_annual_value > csr_net_annual_value:
            recommendation = "American Express Platinum Card"
            net_difference = amex_net_annual_value - csr_net_annual_value
            reason = f"Yields ${net_difference:.2f} higher Net Annual Value based on your spend profile and credit usage."
        else:
            recommendation = "Tie"
            net_difference = 0.0
            reason = "Both cards offer equal Net Annual Value for your specific spend profile."

        return {
            "spend_profile": spend,
            "chase_sapphire_reserve": {
                "annual_fee": self.CSR_ANNUAL_FEE,
                "total_points_earned": total_csr_points,
                "points_dollar_value": csr_points_cash_value,
                "total_credits_value": total_csr_credits_val,
                "credits_breakdown": csr_credits_breakdown,
                "net_annual_value": csr_net_annual_value
            },
            "amex_platinum": {
                "annual_fee": self.AMEX_PLAT_ANNUAL_FEE,
                "total_points_earned": total_amex_points,
                "points_dollar_value": amex_points_cash_value,
                "total_credits_value": total_amex_credits_val,
                "credits_breakdown": amex_credits_breakdown,
                "net_annual_value": amex_net_annual_value
            },
            "recommendation": recommendation,
            "net_difference": net_difference,
            "recommendation_reason": reason
        }

    def format_summary_markdown(self, calc_result: Dict[str, Any]) -> str:
        """Formats financial calculation result into clean Markdown."""
        csr = calc_result["chase_sapphire_reserve"]
        amex = calc_result["amex_platinum"]

        md = (
            f"### Financial Calculation Summary\n\n"
            f"| Metric | Chase Sapphire Reserve | Amex Platinum Card |\n"
            f"|---|---|---|\n"
            f"| **Annual Fee** | -${csr['annual_fee']:,.2f} | -${amex['annual_fee']:,.2f} |\n"
            f"| **Points Earned** | {csr['total_points_earned']:,.0f} pts | {amex['total_points_earned']:,.0f} pts |\n"
            f"| **Points Value (@1.5¢/pt)** | +${csr['points_dollar_value']:,.2f} | +${amex['points_dollar_value']:,.2f} |\n"
            f"| **Statement Credits Value** | +${csr['total_credits_value']:,.2f} | +${amex['total_credits_value']:,.2f} |\n"
            f"| **Net Annual Value** | **${csr['net_annual_value']:,.2f}** | **${amex['net_annual_value']:,.2f}** |\n\n"
        )

        return md
