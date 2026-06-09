"""Genera CSV para subir a Google Sheets con todos los resultados."""
import numpy as np
import math
import pandas as pd
from collections import Counter
from src.data_loader import load_results
from src.strength_hybrid import train_hybrid_model
from src.wc2026 import GROUPS_2026
import csv
import io

MARKET = {
    "Spain": 15.8, "France": 14.9, "England": 10.8,
    "Brazil": 9.1, "Portugal": 9.1, "Argentina": 8.7,
    "Germany": 5.8, "Netherlands": 4.1, "Norway": 2.4,
    "Belgium": 2.1, "Colombia": 2.1, "Japan": 1.9,
    "Croatia": 1.7, "Switzerland": 1.7, "Morocco": 1.7,
    "Mexico": 1.5, "United States": 1.4, "Uruguay": 1.3,
    "Turkey": 1.1, "Ecuador": 0.9, "Senegal": 0.9,
    "South Korea": 0.6, "Canada": 0.4,
}


def main():
    df = load_results()
    cutoff = pd.Timestamp("2026-06-11")
    print("Entrenando modelo...")
    model = train_hybrid_model(df, cutoff)
    rng = np.random.default_rng(42)
    N = 10000

    def plam(h, a):
        sh = model.strength.get(h, 0.0)
        sa = model.strength.get(a, 0.0)
        d = sh - sa
        return math.exp(0.25 + d / model.scale), math.exp(0.25 - d / model.scale)

    # Collect data
    group_results = {}
    for gname, teams in GROUPS_2026.items():
        for i in range(4):
            for j in range(i+1, 4):
                group_results[(gname, teams[i], teams[j])] = []

    group_pos = {g: {t: Counter() for t in teams} for g, teams in GROUPS_2026.items()}
    group_pts = {g: {t: [] for t in teams} for g, teams in GROUPS_2026.items()}
    group_gf = {g: {t: [] for t in teams} for g, teams in GROUPS_2026.items()}
    group_ga = {g: {t: [] for t in teams} for g, teams in GROUPS_2026.items()}
    champion_count = Counter()
    finalist_count = Counter()
    semi_count = Counter()
    qf_count = Counter()
    final_matchup = Counter()

    # KO tracking
    r32_winners_count = [Counter() for _ in range(16)]
    r16_matchups_count = [Counter() for _ in range(8)]
    r16_winners_count = [Counter() for _ in range(8)]
    qf_matchups_count = [Counter() for _ in range(4)]
    qf_winners_count = [Counter() for _ in range(4)]
    sf_matchups_count = [Counter() for _ in range(2)]
    sf_winners_count = [Counter() for _ in range(2)]

    print(f"Simulando {N:,} mundiales...", flush=True)
    for sim in range(N):
        if (sim+1) % 2500 == 0:
            print(f"  {sim+1:,}/{N:,}...", flush=True)

        standings = {}
        for gn, teams in GROUPS_2026.items():
            pts=[0]*4; gf=[0]*4; ga=[0]*4
            for i in range(4):
                for j in range(i+1,4):
                    lh,la = plam(teams[i], teams[j])
                    gh=int(rng.poisson(lh)); ga_=int(rng.poisson(la))
                    group_results[(gn,teams[i],teams[j])].append((gh,ga_))
                    gf[i]+=gh;ga[i]+=ga_;gf[j]+=ga_;ga[j]+=gh
                    if gh>ga_:pts[i]+=3
                    elif gh==ga_:pts[i]+=1;pts[j]+=1
                    else:pts[j]+=3
            idx=list(range(4))
            idx.sort(key=lambda k:(pts[k],gf[k]-ga[k],gf[k],rng.random()),reverse=True)
            standings[gn]=[teams[k] for k in idx]
            for pos,k in enumerate(idx):
                group_pos[gn][teams[k]][pos+1]+=1
                group_pts[gn][teams[k]].append(pts[k])
                group_gf[gn][teams[k]].append(gf[k])
                group_ga[gn][teams[k]].append(ga[k])

        thirds=[standings[g][2] for g in sorted(standings)]
        tl=thirds[:8]
        winners={g:standings[g][0] for g in standings}
        runners={g:standings[g][1] for g in standings}

        r32=[(winners["A"],tl[0]),(runners["C"],runners["D"]),(winners["B"],tl[1]),(runners["E"],runners["F"]),(winners["G"],tl[2]),(runners["I"],runners["J"]),(winners["H"],tl[3]),(runners["K"],runners["L"]),(winners["C"],tl[4]),(runners["A"],runners["B"]),(winners["D"],tl[5]),(runners["G"],runners["H"]),(winners["I"],tl[6]),(winners["F"],runners["I"]),(winners["J"],tl[7]),(winners["L"],runners["K"])]

        def ko(h,a):
            lh,la=plam(h,a)
            gh=int(rng.poisson(lh));ga=int(rng.poisson(la))
            if gh==ga:
                if rng.random()<lh/(lh+la):gh+=1
                else:ga+=1
            return h if gh>ga else a

        r32w=[ko(h,a) for h,a in r32]
        for i,w in enumerate(r32w): r32_winners_count[i][w]+=1

        r16=[(r32w[i],r32w[i+1]) for i in range(0,16,2)]
        for i,(h,a) in enumerate(r16): r16_matchups_count[i][(h,a)]+=1
        r16w=[ko(h,a) for h,a in r16]
        for i,w in enumerate(r16w): r16_winners_count[i][w]+=1
        for t in r16w: qf_count[t]+=1

        qf=[(r16w[i],r16w[i+1]) for i in range(0,8,2)]
        for i,(h,a) in enumerate(qf): qf_matchups_count[i][(h,a)]+=1
        qfw=[ko(h,a) for h,a in qf]
        for i,w in enumerate(qfw): qf_winners_count[i][w]+=1

        sf=[(qfw[0],qfw[1]),(qfw[2],qfw[3])]
        for i,(h,a) in enumerate(sf): sf_matchups_count[i][(h,a)]+=1
        sfw=[ko(h,a) for h,a in sf]
        for i,w in enumerate(sfw): sf_winners_count[i][w]+=1

        champ=ko(sfw[0],sfw[1])
        loser=sfw[1] if champ==sfw[0] else sfw[0]
        champion_count[champ]+=1
        finalist_count[champ]+=1;finalist_count[loser]+=1
        final_matchup[tuple(sorted([sfw[0],sfw[1]]))]+=1
        for t in sfw:semi_count[t]+=1
        for i,(h,a) in enumerate(sf):
            sl=a if sfw[i]==h else h
            semi_count[sl]+=1

    # === BUILD CSV ===
    output = io.StringIO()

    # Sheet 1: CAMPEON
    output.write("MUNDIAL 2026 - PRONOSTICO COMPLETO (10000 simulaciones Monte Carlo)\n")
    output.write("Fecha: 9 de junio de 2026 | Modelo: Hibrido 14 features\n\n")

    output.write("PROBABILIDADES DE CAMPEON\n")
    output.write("#,Equipo,Campeon %,Final %,Semi %,QF %,Mercado %,Diferencia\n")
    for rank,(team,count) in enumerate(champion_count.most_common(20),1):
        pc=count/N*100
        pf=finalist_count.get(team,0)/N*100
        ps=semi_count.get(team,0)/N*100
        pq=qf_count.get(team,0)/N*100
        mkt=MARKET.get(team,0)
        diff=pc-mkt
        output.write(f"{rank},{team},{pc:.1f},{pf:.1f},{ps:.1f},{pq:.1f},{mkt:.1f},{diff:+.1f}\n")

    # Finales
    output.write("\nFINALES MAS PROBABLES\n")
    output.write("Final,Probabilidad %\n")
    for (t1,t2),c in final_matchup.most_common(10):
        output.write(f"{t1} vs {t2},{c/N*100:.1f}\n")

    # Groups
    for gname in sorted(GROUPS_2026.keys()):
        teams = GROUPS_2026[gname]
        output.write(f"\nGRUPO {gname} - TABLA\n")
        output.write("Pos,Equipo,Pts,GF,GC,GD,1ro %,2do %,3ro %,4to %,Clasifica %\n")

        team_data = []
        for t in teams:
            c=group_pos[gname][t]
            p1=c.get(1,0)/N*100;p2=c.get(2,0)/N*100
            p3=c.get(3,0)/N*100;p4=c.get(4,0)/N*100
            pts=np.mean(group_pts[gname][t])
            gf=np.mean(group_gf[gname][t])
            ga=np.mean(group_ga[gname][t])
            team_data.append((t,pts,gf,ga,p1,p2,p3,p4,p1+p2))
        team_data.sort(key=lambda x:-x[8])

        for pos,(t,pts,gf,ga,p1,p2,p3,p4,cl) in enumerate(team_data,1):
            output.write(f"{pos},{t},{pts:.1f},{gf:.1f},{ga:.1f},{gf-ga:+.1f},{p1:.1f},{p2:.1f},{p3:.1f},{p4:.1f},{cl:.1f}\n")

        output.write(f"\nGRUPO {gname} - PARTIDOS\n")
        output.write("Partido,Pronostico,Moda,Gana 1 %,Empate %,Gana 2 %\n")
        for i in range(4):
            for j in range(i+1,4):
                key=(gname,teams[i],teams[j])
                results=group_results[key]
                avg_h=np.mean([r[0] for r in results])
                avg_a=np.mean([r[1] for r in results])
                moda=Counter(results).most_common(1)[0]
                w1=sum(1 for g1,g2 in results if g1>g2)/N*100
                dr=sum(1 for g1,g2 in results if g1==g2)/N*100
                w2=sum(1 for g1,g2 in results if g1<g2)/N*100
                output.write(f"{teams[i]} vs {teams[j]},{avg_h:.1f} - {avg_a:.1f},{moda[0][0]}-{moda[0][1]},{w1:.0f},{dr:.0f},{w2:.0f}\n")

    # Knockout
    r32_desc = ["1roA vs 3ro","2doC vs 2doD","1roB vs 3ro","2doE vs 2doF",
                "1roG vs 3ro","2doI vs 2doJ","1roH vs 3ro","2doK vs 2doL",
                "1roC vs 3ro","2doA vs 2doB","1roD vs 3ro","2doG vs 2doH",
                "1roI vs 3ro","1roF vs 2doI","1roJ vs 3ro","1roL vs 2doK"]

    output.write("\nRONDA DE 32\n")
    output.write("#,Estructura,Avanza (mas probable),Prob %\n")
    for slot in range(16):
        top = r32_winners_count[slot].most_common(1)
        if top:
            w, wc = top[0]
            output.write(f"{slot+1},{r32_desc[slot]},{w},{wc/N*100:.1f}\n")

    output.write("\nOCTAVOS DE FINAL\n")
    output.write("#,Partido mas probable,Prob matchup %,Avanza,Prob avanza %\n")
    for slot in range(8):
        top_m = r16_matchups_count[slot].most_common(1)
        top_w = r16_winners_count[slot].most_common(1)
        if top_m and top_w:
            (h,a),mc = top_m[0]
            w,wc = top_w[0]
            output.write(f"{slot+1},{h} vs {a},{mc/N*100:.1f},{w},{wc/N*100:.1f}\n")

    output.write("\nCUARTOS DE FINAL\n")
    output.write("#,Partido mas probable,Prob matchup %,Avanza,Prob avanza %\n")
    for slot in range(4):
        top_m = qf_matchups_count[slot].most_common(1)
        top_w = qf_winners_count[slot].most_common(1)
        if top_m and top_w:
            (h,a),mc = top_m[0]
            w,wc = top_w[0]
            output.write(f"{slot+1},{h} vs {a},{mc/N*100:.1f},{w},{wc/N*100:.1f}\n")

    output.write("\nSEMIFINALES\n")
    output.write("#,Partido mas probable,Prob matchup %,Avanza,Prob avanza %\n")
    for slot in range(2):
        top_m = sf_matchups_count[slot].most_common(1)
        top_w = sf_winners_count[slot].most_common(1)
        if top_m and top_w:
            (h,a),mc = top_m[0]
            w,wc = top_w[0]
            output.write(f"{slot+1},{h} vs {a},{mc/N*100:.1f},{w},{wc/N*100:.1f}\n")

    output.write("\nFINAL Y CAMPEON\n")
    output.write("Final mas probable,Prob %\n")
    t1t2, fc = final_matchup.most_common(1)[0]
    output.write(f"{t1t2[0]} vs {t1t2[1]},{fc/N*100:.1f}\n")
    output.write(f"\nCampeon mas probable,Prob %\n")
    champ, cc = champion_count.most_common(1)[0]
    output.write(f"{champ},{cc/N*100:.1f}\n")

    # Methodology
    output.write("\nMETODOLOGIA\n")
    output.write("Aspecto,Detalle\n")
    output.write("Simulaciones,10000 mundiales via Monte Carlo\n")
    output.write("Distribucion de goles,Poisson (lambdas derivadas de fuerza)\n")
    output.write("Features Layer 1,Elo (50%) + Value (11%) + Liga (7%) + Momentum (14%) + Defensa (18%) + Campeon (8%)\n")
    output.write("Features Layer 2,Mercado (32%) + Edad (5%) + DT (4%) + Host (2%) + Poblacion (2%) + Defensa2 (6%) + Diversidad (3%) + Composicion (3%) + Frontrunner (5%)\n")
    output.write("Elo corregido,Time decay 0.94/anio + calidad rival + K clasificatorias reducido\n")
    output.write("Liga corregida,Descuento domestico 40% (ingles en PL vs marroqui en PL)\n")
    output.write("Validacion,Backtest 2010-2022: Brier 0.043 vs mercado 0.045 (modelo mejor)\n")
    output.write("Correlacion con mercado,Spearman 0.836 - Pearson 0.925\n")

    csv_content = output.getvalue()

    with open("mundial_2026_completo.csv", "w", encoding="utf-8") as f:
        f.write(csv_content)

    print(f"\nCSV guardado: mundial_2026_completo.csv ({len(csv_content)} chars)")
    return csv_content


if __name__ == "__main__":
    main()
