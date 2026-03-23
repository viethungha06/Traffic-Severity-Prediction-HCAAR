import pandas as pd
import folium
from folium.plugins import MarkerCluster
import os

def create_traffic_risk_map(
    csv_path='output/cleaned_dataset.csv',
    output_map='output/Traffic_Risk_Map.html'
):
    # Step 1: Load cleaned dataset
    print("[*] Reading data...")
    if not os.path.exists(csv_path):
        print("[!] File not found. Please run cleaning.py first!")
        return

    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=['Latitude', 'Longitude', 'Severity_Class'])

    center_lat = df['Latitude'].median()
    center_lon = df['Longitude'].median()

    # Step 2: Initialize satellite base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles=None
    )

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='ESRI World Imagery',
        name='Satellite (ESRI)',
        overlay=False,
        control=True,
        show=True
    ).add_to(m)

    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Standard Map',
        overlay=False,
        control=True,
        show=False
    ).add_to(m)

    # Step 3: Sample data by severity class
    df3 = df[df['Severity_Class'] == 3].dropna(subset=['Latitude', 'Longitude'])
    df2 = df[df['Severity_Class'] == 2].dropna(subset=['Latitude', 'Longitude'])
    df2 = df2.sample(n=min(3000, len(df2)), random_state=42)

    print(f"[*] Class 3 (Severe/Fatal): {len(df3):,} points")
    print(f"[*] Class 2 (Minor):        {len(df2):,} points")

    # Step 4: Add Class 3 layer — Severe/Fatal — dark red, large markers
    layer3 = folium.FeatureGroup(name="Class 3 - Severe/Fatal (red)", show=True)
    cluster3 = MarkerCluster(
        name="cluster3",
        overlay=True,
        control=False,
        options={'maxClusterRadius': 50, 'disableClusteringAtZoom': 14}
    ).add_to(layer3)

    for _, row in df3.iterrows():
        speed    = int(row.get('Speed Limit', 35))
        col_type = str(row.get('Collision Type', 'Unknown'))
        weather  = str(row.get('Weather_Group', 'Unknown'))
        hour     = int(row['Hour']) if 'Hour' in df3.columns else 'N/A'

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=9,
            color='white',
            weight=2,
            fill=True,
            fill_color='#cc0000',
            fill_opacity=0.95,
            popup=folium.Popup(
                f"<b style='color:#cc0000'>Severe / Fatal</b><br>"
                f"Hour: {hour}h | Speed: {speed} mph<br>"
                f"Collision: {col_type}<br>"
                f"Weather: {weather}",
                max_width=210
            ),
            tooltip=f"Severe/Fatal | {speed} mph | {col_type}"
        ).add_to(cluster3)

    layer3.add_to(m)

    # Step 5: Add Class 2 layer — Minor — orange, small markers
    layer2 = folium.FeatureGroup(name="Class 2 - Minor/Moderate (orange)", show=True)
    cluster2 = MarkerCluster(
        name="cluster2",
        overlay=True,
        control=False,
        options={'maxClusterRadius': 60, 'disableClusteringAtZoom': 15}
    ).add_to(layer2)

    for _, row in df2.iterrows():
        speed    = int(row.get('Speed Limit', 35))
        col_type = str(row.get('Collision Type', 'Unknown'))

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color='white',
            weight=1,
            fill=True,
            fill_color='#e07b00',
            fill_opacity=0.75,
            popup=folium.Popup(
                f"<b style='color:#e07b00'>Minor / Moderate</b><br>"
                f"Speed: {speed} mph<br>"
                f"Collision: {col_type}",
                max_width=190
            ),
            tooltip=f"Minor | {speed} mph"
        ).add_to(cluster2)

    layer2.add_to(m)

    # Step 6: Add title and legend HTML overlays
    title_html = """
    <div style="
        position:fixed; top:15px; left:50%; transform:translateX(-50%);
        z-index:1000; background:rgba(0,0,0,0.82);
        color:white; padding:10px 24px; border-radius:8px;
        font-family:Arial; font-size:14px; font-weight:bold;
        border:1px solid #ff4444; white-space:nowrap;">
        US Traffic Accident Risk Map &mdash; Montgomery County, MD (2015&ndash;2026)
    </div>
    """
    legend_html = """
    <div style="
        position:fixed; bottom:30px; left:20px; z-index:1000;
        background:rgba(0,0,0,0.82); color:white;
        padding:12px 16px; border-radius:8px;
        font-family:Arial; font-size:12px; line-height:2.0;">
        <b>Legend</b><br>
        <svg width="12" height="12"><circle cx="6" cy="6" r="6" fill="#cc0000"/></svg>
        &nbsp;Class 3 &mdash; Severe/Fatal (red, large)<br>
        <svg width="8" height="8"><circle cx="4" cy="4" r="4" fill="#e07b00"/></svg>
        &nbsp;Class 2 &mdash; Minor/Moderate (orange, small)<br>
        <hr style="margin:6px 0; border-color:#555;">
        <span style="color:#aaa; font-size:11px;">
            Zoom in to expand clusters &rarr; click a point for details
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(legend_html))

    # Step 7: Add layer control (always expanded, top-right)
    folium.LayerControl(
        collapsed=False,
        position='topright'
    ).add_to(m)

    # Step 8: Save map to HTML file
    os.makedirs(os.path.dirname(output_map), exist_ok=True)
    m.save(output_map)

    print(f"\n[V] SUCCESS! Map exported to: {output_map}")
    print(f"    Class 3: {len(df3):,} points | Class 2: {len(df2):,} points")
    print("    Open the HTML file in Chrome/Edge to view!")

if __name__ == "__main__":
    create_traffic_risk_map()