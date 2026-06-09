"""World Cup 2026 groups and bracket structure.

48 teams, 12 groups of 4.
Top 2 per group (24) + 8 best 3rd-placed teams = 32 advance to knockout.
Knockout: R32 -> R16 -> QF -> SF -> 3rd place -> Final.

Official draw: December 5, 2025, Washington D.C.
Hosts: Mexico (A1), Canada (B1), United States (D1).
"""

GROUPS_2026 = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Cura\u00e7ao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Colombia", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def build_r32(winners, runners, thirds_8):
    """Build R32 bracket. All 12 winners, 12 runners, 8 thirds used exactly once.

    Args:
        winners: dict group -> 1st place team
        runners: dict group -> 2nd place team
        thirds_8: list of 8 best 3rd-place teams (assigned to slots)

    Returns:
        list of 16 (home, away) tuples for R32
    """
    tl = thirds_8
    return [
        # Left half -> QF1, QF2 -> SF1
        (winners["A"], tl[0]),        # R32-1:  1A vs 3rd
        (runners["E"], runners["F"]), # R32-2:  2E vs 2F
        (winners["B"], tl[1]),        # R32-3:  1B vs 3rd
        (winners["E"], runners["L"]), # R32-4:  1E vs 2L
        (winners["G"], tl[2]),        # R32-5:  1G vs 3rd
        (runners["I"], runners["J"]), # R32-6:  2I vs 2J
        (winners["H"], tl[3]),        # R32-7:  1H vs 3rd
        (winners["F"], runners["K"]), # R32-8:  1F vs 2K
        # Right half -> QF3, QF4 -> SF2
        (winners["C"], tl[4]),        # R32-9:  1C vs 3rd
        (runners["A"], runners["B"]), # R32-10: 2A vs 2B
        (winners["D"], tl[5]),        # R32-11: 1D vs 3rd
        (runners["G"], runners["H"]), # R32-12: 2G vs 2H
        (winners["I"], tl[6]),        # R32-13: 1I vs 3rd
        (winners["K"], runners["D"]), # R32-14: 1K vs 2D
        (winners["J"], tl[7]),        # R32-15: 1J vs 3rd
        (winners["L"], runners["C"]), # R32-16: 1L vs 2C
    ]


R32_DESCRIPTIONS = [
    "1A vs 3ro", "2E vs 2F", "1B vs 3ro", "1E vs 2L",
    "1G vs 3ro", "2I vs 2J", "1H vs 3ro", "1F vs 2K",
    "1C vs 3ro", "2A vs 2B", "1D vs 3ro", "2G vs 2H",
    "1I vs 3ro", "1K vs 2D", "1J vs 3ro", "1L vs 2C",
]
