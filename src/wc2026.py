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
    """Build R32 bracket per official FIFA 2026 knockout structure.

    Official match numbers: M73-M88 (R32), M89-M96 (R16), M97-M100 (QF).
    Bracket: consecutive pairs feed into same R16 match.
    R16[0]=R32[0]vsR32[1], R16[1]=R32[2]vsR32[3], etc.
    QF[0]=R16[0]vsR16[1], QF[1]=R16[2]vsR16[3], etc.

    Fixed pairings (8 winner-vs-3rd, 4 winner-vs-runner, 4 runner-vs-runner):
      M73: 2A vs 2B     M77: 1I vs 3rd    M74: 1E vs 3rd    M75: 1C vs 2F
      M76: 1F vs 2C     M78: 2E vs 2I     M79: 1A vs 3rd    M80: 1D vs 3rd
      M81: 1G vs 3rd    M82: 1L vs 3rd    M83: 1B vs 3rd    M84: 1K vs 3rd
      M85: 1H vs 2J     M86: 1J vs 2H     M87: 2D vs 2G     M88: 2K vs 2L

    Third-place slot assignment depends on which 8 groups produce the best
    3rds (495 FIFA scenarios). Simplified here: assigned in bracket order.

    Args:
        winners: dict group -> 1st place team
        runners: dict group -> 2nd place team
        thirds_8: list of 8 best 3rd-place teams (assigned to slots)

    Returns:
        list of 16 (home, away) tuples for R32
    """
    tl = thirds_8
    return [
        # --- Left half -> QF1, QF2 -> SF1 ---
        # R16-1 (M89): W73 vs W77
        (runners["A"], runners["B"]),    # M73: 2A vs 2B
        (winners["I"], tl[0]),           # M77: 1I vs 3rd
        # R16-2 (M90): W74 vs W75
        (winners["E"], tl[1]),           # M74: 1E vs 3rd
        (winners["C"], runners["F"]),    # M75: 1C vs 2F
        # R16-3 (M91): W76 vs W78
        (winners["F"], runners["C"]),    # M76: 1F vs 2C
        (runners["E"], runners["I"]),    # M78: 2E vs 2I
        # R16-4 (M92): W79 vs W80
        (winners["A"], tl[2]),           # M79: 1A vs 3rd
        (winners["D"], tl[3]),           # M80: 1D vs 3rd
        # --- Right half -> QF3, QF4 -> SF2 ---
        # R16-5 (M93): W81 vs W82
        (winners["G"], tl[4]),           # M81: 1G vs 3rd
        (winners["L"], tl[5]),           # M82: 1L vs 3rd
        # R16-6 (M94): W83 vs W84
        (winners["B"], tl[6]),           # M83: 1B vs 3rd
        (winners["K"], tl[7]),           # M84: 1K vs 3rd
        # R16-7 (M95): W85 vs W86
        (winners["H"], runners["J"]),    # M85: 1H vs 2J
        (winners["J"], runners["H"]),    # M86: 1J vs 2H
        # R16-8 (M96): W87 vs W88
        (runners["D"], runners["G"]),    # M87: 2D vs 2G
        (runners["K"], runners["L"]),    # M88: 2K vs 2L
    ]


R32_DESCRIPTIONS = [
    "2A vs 2B", "1I vs 3ro", "1E vs 3ro", "1C vs 2F",
    "1F vs 2C", "2E vs 2I", "1A vs 3ro", "1D vs 3ro",
    "1G vs 3ro", "1L vs 3ro", "1B vs 3ro", "1K vs 3ro",
    "1H vs 2J", "1J vs 2H", "2D vs 2G", "2K vs 2L",
]
