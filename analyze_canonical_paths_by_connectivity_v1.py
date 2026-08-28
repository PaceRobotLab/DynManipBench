from pathlib import Path
import csv, json, zipfile, re
import numpy as np

ROOT = Path.home() / "DynManipBench"
PATH_ZIP = ROOT / "data" / "raw" / "pathData.zip"
CONN_DIR = ROOT / "runs" / "dataset_endpoint_connectivity_full_v1" / "environments"
OUT = ROOT / "runs" / "canonical_path_connectivity_postanalysis_v1"
OUT.mkdir(parents=True, exist_ok=True)

CONT = np.array([0,2,4,6], dtype=int)

def parse_configs(data):
    rows=[]
    for line in data.decode("utf-8", errors="replace").splitlines():
        s=line.strip()
        if not s or s.startswith("#"): continue
        try:
            vals=[float(x) for x in s.replace(","," ").split()]
        except ValueError:
            continue
        if len(vals)>=10 and int(round(vals[0]))==len(vals)-1:
            vals=vals[1:]
        if len(vals) in (7,9):
            rows.append(vals)
    a=np.asarray(rows,float)
    if a.ndim==2 and a.shape[1]==9:
        a=a[:,1:8]
    return a

def wd(a,b):
    d=np.asarray(b)-np.asarray(a)
    d=d.copy()
    d[CONT]=np.arctan2(np.sin(d[CONT]),np.cos(d[CONT]))
    return float(np.linalg.norm(d))

def cost(q):
    return sum(wd(q[i],q[i+1]) for i in range(len(q)-1))

def load_connectivity():
    """Load all per-environment connectivity rows and index by env/path_row."""
    idx={}
    files=sorted(CONN_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No per-environment CSVs under {CONN_DIR}")
    for p in files:
        with p.open(newline="") as f:
            for r in csv.DictReader(f):
                env=int(r["environment"])
                row=int(r["path_row"])
                idx[(env,row)] = {
                    "block":r["block"],
                    "family":int(r["family"]),
                    "direct_free":str(r["direct_free"]).lower() in ("1","true","yes"),
                    "valid_endpoints":str(r["valid_endpoints"]).lower() in ("1","true","yes"),
                    "first_collision_fraction":float(r["first_collision_fraction"]) if r.get("first_collision_fraction") not in ("",None,"nan") else None,
                }
    return idx

def archive_map(z):
    """Map env/path index candidates from archive names."""
    result={}
    pat=re.compile(r'env(\d+)[_/](?:env\d+_)?(\d+)\.configs$', re.I)
    pat2=re.compile(r'env(\d+)_(\d+)\.configs$', re.I)
    for n in z.namelist():
        m=pat.search(n) or pat2.search(Path(n).name)
        if m:
            env=int(m.group(1)); k=int(m.group(2))
            result[(env,k)]=n
    return result

def quant(a):
    a=np.asarray(a,float)
    return {k:float(v) for k,v in zip(
        ["min","q05","q25","median","mean","q75","q95","max"],
        [a.min(),np.quantile(a,.05),np.quantile(a,.25),np.median(a),a.mean(),
         np.quantile(a,.75),np.quantile(a,.95),a.max()])}

def main():
    conn=load_connectivity()
    print(f"Loaded connectivity labels: {len(conn):,}")
    if len(conn)!=750000:
        print("WARNING: expected 750,000 connectivity rows.")

    records=[]
    with zipfile.ZipFile(PATH_ZIP) as z:
        amap=archive_map(z)
        print(f"Indexed archive .configs: {len(amap):,}")
        for n,(key,c) in enumerate(conn.items(),1):
            env,row=key
            # Try both 0-based and 1-based archive indexing.
            member=amap.get((env,row))
            if member is None:
                member=amap.get((env,row+1))
            if member is None:
                raise KeyError(f"Could not locate archive trajectory env={env} row={row}")
            q=parse_configs(z.read(member))
            if q.ndim!=2 or q.shape[1]!=7 or len(q)<2:
                raise ValueError(f"Bad trajectory {member}: shape={q.shape}")
            ed=wd(q[0],q[-1])
            pc=cost(q)
            ratio=pc/ed if ed>1e-12 else np.nan
            records.append({
                "environment":env, "family":c["family"], "block":c["block"],
                "path_row":row, "archive_member":member,
                "direct_free":int(c["direct_free"]),
                "canonical_waypoints":len(q),
                "endpoint_distance_rad":ed,
                "canonical_path_cost_rad":pc,
                "canonical_cost_direct_ratio":ratio,
            })
            if n%50000==0:
                print(f"Processed {n:,}/750,000")

    outcsv=OUT/"canonical_path_connectivity.csv"
    fields=list(records[0])
    with outcsv.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(records)

    def group(rows):
        return {
            "n":len(rows),
            "waypoints":quant([r["canonical_waypoints"] for r in rows]),
            "endpoint_distance_rad":quant([r["endpoint_distance_rad"] for r in rows]),
            "path_cost_rad":quant([r["canonical_path_cost_rad"] for r in rows]),
            "cost_direct_ratio":quant([r["canonical_cost_direct_ratio"] for r in rows
                                      if np.isfinite(r["canonical_cost_direct_ratio"])]),
        }

    direct=[r for r in records if r["direct_free"]]
    blocked=[r for r in records if not r["direct_free"]]
    summary={"overall":{"direct":group(direct),"blocked":group(blocked)},"by_block":{}}
    for b in "ABC":
        br=[r for r in records if r["block"]==b]
        summary["by_block"][b]={
            "direct":group([r for r in br if r["direct_free"]]),
            "blocked":group([r for r in br if not r["direct_free"]])
        }

    js=OUT/"canonical_path_connectivity_summary.json"
    js.write_text(json.dumps(summary,indent=2))

    print("\nCANONICAL PATH / CONNECTIVITY SUMMARY")
    print("="*80)
    for label,rows in [("DIRECT",direct),("BLOCKED",blocked)]:
        g=group(rows)
        print(f"{label}: n={g['n']:,}")
        print(f"  waypoints median={g['waypoints']['median']:.1f}, mean={g['waypoints']['mean']:.3f}")
        print(f"  endpoint median={g['endpoint_distance_rad']['median']:.6f}")
        print(f"  path cost median={g['path_cost_rad']['median']:.6f}")
        print(f"  cost/direct median={g['cost_direct_ratio']['median']:.6f}, mean={g['cost_direct_ratio']['mean']:.6f}")
    print("\nBY BLOCK")
    for b in "ABC":
        d=summary["by_block"][b]["direct"]; x=summary["by_block"][b]["blocked"]
        print(f"{b}: direct wp med={d['waypoints']['median']:.1f}, ratio med={d['cost_direct_ratio']['median']:.4f} | "
              f"blocked wp med={x['waypoints']['median']:.1f}, ratio med={x['cost_direct_ratio']['median']:.4f}")
    print("\nOUTPUT")
    print(outcsv); print(js)
    print("\nPASS — canonical archive analysis completed.")

if __name__=="__main__":
    main()
