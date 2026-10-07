from __future__ import annotations


def format_table(sim_rows: list[dict], stag_rows: list[dict] | None = None) -> str:
    lines = [
        ,
         * 88,
    ]
    stag_by_fleet = {r["fleet"]: r for r in (stag_rows or [])}
    for row in sim_rows:
        stag = stag_by_fleet.get(row["fleet"])
        stag_txt = ""
        if stag is not None:
            stag_txt = f"{stag['crash_pct']:4.1f} [{stag['crash_lo']:4.1f}, {stag['crash_hi']:4.1f}]"
        lines.append(


        )
    if not sim_rows and stag_rows:
        lines = ["Fleet  Hierarchy    Stag crash", "-" * 40]
        for stag in stag_rows:
            lines.append(


            )
    return "\n".join(lines)
