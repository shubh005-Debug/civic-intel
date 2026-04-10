"""Red Alert Zone Detection using Pandas + spatial clustering"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

THRESHOLD     = 3       # min critical complaints to trigger alert
RADIUS_DEG    = 0.02    # ~2 km radius in degrees
TIME_WINDOW_H = 72      # hours

def detect_red_alert_zones(complaints: list) -> list:
    """
    Given a list of complaint dicts, return list of red alert zones.
    Each zone: {lat, lng, complaint_count, area_name, complaints:[ids]}
    """
    if not complaints:
        return []

    df = pd.DataFrame(complaints)
    df['created_at'] = pd.to_datetime(df['created_at'])
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=TIME_WINDOW_H)
    critical = df[(df['priority'] == 'Critical') & (df['is_fake'] == 0) & (df['created_at'] >= cutoff)].copy()

    if critical.empty:
        return []

    zones = []
    visited = set()

    for idx, row in critical.iterrows():
        if idx in visited:
            continue
        lat, lng = row['latitude'], row['longitude']
        mask = (
            (np.abs(critical['latitude']  - lat) <= RADIUS_DEG) &
            (np.abs(critical['longitude'] - lng) <= RADIUS_DEG)
        )
        cluster = critical[mask]
        ids = list(cluster.index)
        if len(cluster) >= THRESHOLD:
            for i in ids: visited.add(i)
            zones.append({
                'latitude':        float(cluster['latitude'].mean()),
                'longitude':       float(cluster['longitude'].mean()),
                'complaint_count': int(len(cluster)),
                'area_name':       row.get('area_name', 'Unknown Area'),
                'complaint_ids':   list(cluster['complaint_id']) if 'complaint_id' in cluster else [],
                'categories':      list(cluster['category'].unique()),
            })

    return zones
