import streamlit as st
import requests
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from datetime import datetime, timezone
import math
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE LA APP WEB ---
st.set_page_config(page_title="Análisis Sinóptico México", layout="wide")
RUTA_SHAPEFILE = "00ent.shp" 
FIRMA_ELABORADO_POR = "3a. Generación Maestría en Ciencias en Meteorología"
# -----------------------------------

# --- FUNCIONES DE AYUDA METEOROLÓGICA ---
def grados_a_cardinal(grados):
    if grados is None or str(grados).upper() == 'VRB': return "VRB"
    try:
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        ix = int((float(grados) + 11.25)/22.5)
        return dirs[ix % 16]
    except: return str(grados)

def calcular_humedad(t, td):
    if t is None or td is None: return "N/D"
    try:
        t_float, td_float = float(t), float(td)
        e = 6.11 * 10.0 ** (7.5 * td_float / (237.3 + td_float))
        es = 6.11 * 10.0 ** (7.5 * t_float / (237.3 + t_float))
        return int(round((e / es) * 100))
    except: return "--"

def obtener_cobertura_maxima(clouds):
    if not clouds: return 'SKC'
    niveles = {'OVC': 4, 'BKN': 3, 'SCT': 2, 'FEW': 1, 'CLR': 0, 'SKC': 0, 'CAVOK': 0}
    max_nivel, max_cov = -1, 'SKC'
    for c in clouds:
        cov = 'SKC' if c.get('cover', 'SKC') in ['CLR', 'CAVOK'] else c.get('cover', 'SKC')
        nivel = niveles.get(cov, 0)
        if nivel > max_nivel: max_nivel, max_cov = nivel, cov
    return max_cov

def obtener_color_reglas_vuelo(vis_raw, nubes):
    vis_sm = 99.0 
    if vis_raw is not None:
        try:
            v_str = str(vis_raw).replace('+', '').replace('V', '').replace('SM', '').strip()
            if '/' in v_str:
                if ' ' in v_str: 
                    entero, frac = v_str.split(' ')
                    num, den = frac.split('/')
                    vis_sm = float(entero) + (float(num)/float(den))
                else:
                    num, den = v_str.split('/')
                    vis_sm = float(num)/float(den)
            else:
                vis_sm = float(v_str)
        except: pass
    
    techo_ft = 99999
    if nubes:
        for c in nubes:
            if c.get('cover') in ['BKN', 'OVC', 'VV']:
                base = c.get('base', 99999)
                if base < techo_ft: techo_ft = base
                
    if techo_ft < 500 or vis_sm < 1.0: return '#d63384' 
    elif techo_ft < 1000 or vis_sm < 3.0: return '#dc3545' 
    elif techo_ft <= 3000 or vis_sm <= 5.0: return '#007bff' 
    else: return '#28a745' 

# --- DICCIONARIOS ---
estaciones_metar = {
    'Aguascalientes': ['MMAS'], 'Baja California': ['MMLT', 'MMTJ'],
    'Baja California Sur': ['MMLP', 'MMSD'], 'Campeche': ['MMCP', 'MMEQ'], 
    'Coahuila de Zaragoza': ['MMTC'], 'Colima': ['MMIA', 'MMZJ'],
    'Chiapas': ['MMTG', 'MMTP'], 'Chihuahua': ['MMCU', 'MMCS'], 'Ciudad de México': ['MMMX'],
    'Durango': ['MMDO'], 'Guanajuato': ['MMLO'], 'Guerrero': ['MMAA', 'MMZH'],
    'Hidalgo': ['MMPA'], 'Jalisco': ['MMGL', 'MMPR'], 'México': ['MMTO'],
    'Michoacán de Ocampo': ['MMMM', 'MMLM'], 'Morelos': ['MMCB'], 'Nayarit': ['MMEP'],
    'Nuevo León': ['MMMY', 'MMAN'], 'Oaxaca': ['MMOX', 'MMPS'], 'Puebla': ['MMPB'],
    'Querétaro': ['MMQT'], 'Quintana Roo': ['MMUN', 'MMCZ'], 'San Luis Potosí': ['MMSP'],
    'Sinaloa': ['MMCL', 'MMMZ'], 'Sonora': ['MMHO', 'MMGY'], 'Tabasco': ['MMVA'],
    'Tamaulipas': ['MMTM', 'MMMA'], 'Tlaxcala': ['MMTL'], 'Veracruz de Ignacio de la Llave': ['MMVR', 'MMMT'],
    'Yucatán': ['MMMD', 'MMKA'], 'Zacatecas': ['MMZC']
}

nombres_aeropuertos = {
    'MMAS': 'Aguascalientes', 'MMLT': 'Loreto', 'MMTJ': 'Tijuana', 'MMLP': 'La Paz', 
    'MMSD': 'San José del Cabo', 'MMCP': 'Campeche', 'MMEQ': 'Ciudad del Carmen', 
    'MMTC': 'Torreón', 'MMIA': 'Colima', 'MMZJ': 'Manzanillo', 'MMTG': 'Tuxtla Gutiérrez', 
    'MMTP': 'Tapachula', 'MMCU': 'Chihuahua', 'MMCS': 'Ciudad Juárez', 'MMMX': 'Ciudad de México',
    'MMDO': 'Durango', 'MMLO': 'León/Bajío', 'MMAA': 'Acapulco', 'MMZH': 'Zihuatanejo', 
    'MMPA': 'Pachuca', 'MMGL': 'Guadalajara', 'MMPR': 'Puerto Vallarta', 'MMTO': 'Toluca', 
    'MMMM': 'Morelia', 'MMLM': 'Los Mochis', 'MMCB': 'Cuernavaca', 'MMEP': 'Tepic', 
    'MMMY': 'Monterrey', 'MMAN': 'Del Norte (Mty)', 'MMOX': 'Oaxaca', 'MMPS': 'Puerto Escondido',
    'MMPB': 'Puebla', 'MMQT': 'Querétaro', 'MMUN': 'Cancún', 'MMCZ': 'Cozumel', 
    'MMSP': 'San Luis Potosí', 'MMCL': 'Culiacán', 'MMMZ': 'Mazatlán', 'MMHO': 'Hermosillo', 
    'MMGY': 'Guaymas', 'MMVA': 'Villahermosa', 'MMTM': 'Tampico', 'MMMA': 'Matamoros',
    'MMTL': 'Tlaxcala', 'MMVR': 'Veracruz', 'MMMT': 'Minatitlán', 'MMMD': 'Mérida', 
    'MMKA': 'Kaua/Chichén Itzá', 'MMZC': 'Zacatecas'
}

lista_todas_estaciones = [icao for sublist in estaciones_metar.values() for icao in sublist]
icaos = ",".join(lista_todas_estaciones)

# --- DESCARGA DE DATOS ---
@st.cache_data(ttl=300)
def cargar_datos():
    url_metar = f"https://aviationweather.gov/api/data/metar?ids={icaos}&format=json"
    url_taf = f"https://aviationweather.gov/api/data/taf?ids={icaos}&format=json"
    
    res_metar = requests.get(url_metar).json()
    res_taf = requests.get(url_taf).json() if requests.get(url_taf).status_code == 200 else []
    
    tafs_dict = {t['icaoId']: t.get('rawTAF', t.get('rawTaf', 'No TAF')) for t in res_taf}
    
    datos_estaciones = []
    clima_por_icao = {}
    hora_gen = "N/D"
    
    if res_metar:
        dt_obs_gen = datetime.fromtimestamp(res_metar[0].get('obsTime', 0), timezone.utc)
        hora_gen = dt_obs_gen.strftime("%d-%b-%Y %H:%M Z")

        for m in res_metar:
            dt_obs = datetime.fromtimestamp(m.get('obsTime', 0), timezone.utc)
            hora_z = dt_obs.strftime("%d-%b-%Y %H:%M Z")
            
            nubes_arr = m.get('clouds', [])
            cobertura = obtener_cobertura_maxima(nubes_arr)
            color_borde = obtener_color_reglas_vuelo(m.get('visib'), nubes_arr)

            # Convertir QNH a hPa para las isobaras (1 inHg = 33.8639 hPa)
            presion_hpa = None
            if m.get('altim'):
                try: presion_hpa = float(m.get('altim')) * 33.8639
                except: pass

            clima_por_icao[m['icaoId']] = {
                'ICAO': m['icaoId'], 'lat': m.get('lat'), 'lon': m.get('lon'),
                'obsTime': hora_z, 'temp': m.get('temp'), 'dewp': m.get('dewp'),
                'wdir': m.get('wdir'), 'wspd': m.get('wspd'), 'wgst': m.get('wgst'),
                'altim': m.get('altim'), 'presion_hpa': presion_hpa, 'visib': m.get('visib'), 
                'cloud_cover': cobertura, 'border_color': color_borde 
            }
            if m.get('lat') and m.get('lon'):
                datos_estaciones.append(clima_por_icao[m['icaoId']])
                
    datos_estado = []
    for estado, lista_icaos in estaciones_metar.items():
        temps = [clima_por_icao[icao]['temp'] for icao in lista_icaos if icao in clima_por_icao and clima_por_icao[icao]['temp'] is not None]
        promedio = sum(float(t) for t in temps) / len(temps) if temps else None
        datos_estado.append({'NOMGEO': estado, 'Temp_Superficie': promedio})
        
    return pd.DataFrame(datos_estado), pd.DataFrame(datos_estaciones), tafs_dict, hora_gen

with st.spinner('Conectando con Aviation Weather Center y Satélite GOES...'):
    df_estados, df_puntos, tafs_dict, hora_formateada = cargar_datos()
    mapa_mexico = gpd.read_file(RUTA_SHAPEFILE).to_crs(epsg=4326)
    mapa_temperatura = mapa_mexico.merge(df_estados, on='NOMGEO', how='left')

# --- CONSTRUCCIÓN DEL MAPA ---
mapa_interactivo = folium.Map(location=[23.6345, -102.5528], zoom_start=5, tiles=None)

# 1. Capas Base
folium.TileLayer('cartodbpositron', name='Mapa Claro', control=True).add_to(mapa_interactivo)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri', name='Satélite Real', overlay=False, control=True
).add_to(mapa_interactivo)

# 2. CAPA GOES-16 EN VIVO (Infrarrojo de Iowa Environmental Mesonet)
folium.raster_layers.WmsTileLayer(
    url='https://mesonet.agron.iastate.edu/cgi-bin/wms/goes/conus_ir.cgi?',
    layers='goes_conus_ir',
    name='Satélite Nubes (GOES-16 IR)',
    fmt='image/png',
    transparent=True,
    overlay=True,
    control=True,
    opacity=0.6
).add_to(mapa_interactivo)

# 3. Capa Mapa de Calor
folium.Choropleth(
    geo_data=mapa_temperatura, name='Temperatura por Estado', data=mapa_temperatura,
    columns=['NOMGEO', 'Temp_Superficie'], key_on='feature.properties.NOMGEO',
    fill_color='RdYlBu_r', fill_opacity=0.5, line_opacity=0.4,
    legend_name='Temperatura Promedio (°C)', nan_fill_color='white'
).add_to(mapa_interactivo)

# 4. CAPA DE ISOBARAS (Interpolación Matemática Scipy/Matplotlib)
capa_isobaras = folium.FeatureGroup(name='Isobaras de Presión (hPa)', show=False)
try:
    df_iso = df_puntos.dropna(subset=['lat', 'lon', 'presion_hpa']).copy()
    if len(df_iso) > 10:
        x, y = df_iso['lon'].values, df_iso['lat'].values
        z = df_iso['presion_hpa'].values
        
        xi = np.linspace(x.min() - 2, x.max() + 2, 200)
        yi = np.linspace(y.min() - 2, y.max() + 2, 200)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolación Cúbica
        zi = griddata((x, y), z, (xi, yi), method='cubic')
        
        # Generar Contornos Virtuales
        fig, ax = plt.subplots()
        niveles_presion = np.arange(int(np.nanmin(zi)), int(np.nanmax(zi)) + 1, 2)
        CS = ax.contour(xi, yi, zi, levels=niveles_presion)
        
        # Extraer líneas y pasarlas a Folium
        for i, collection in enumerate(CS.collections):
            val = CS.levels[i]
            for path in collection.get_paths():
                v = path.vertices
                coords = [[lat, lon] for lon, lat in v]
                folium.PolyLine(
                    coords, color='black', weight=2, opacity=0.8,
                    tooltip=f"<b>{int(val)} hPa</b>"
                ).add_to(capa_isobaras)
        plt.close(fig)
except Exception as e:
    print(f"Error generando isobaras: {e}")
capa_isobaras.add_to(mapa_interactivo)

# --- SOBREPOSICIONES HTML/CSS ---
consolidated_overlays = f"""
    <style>.leaflet-top {{ top: 75px !important; }}</style>
    <div style="width: 100%; position: absolute; top: 0; left: 0; background-color: #8B0000; color: white; padding: 10px 0; text-align: center; font-family: Arial, sans-serif; font-size: 20px; font-weight: bold; z-index: 9999; box-shadow: 0 2px 5px rgba(0,0,0,0.5);">
        Análisis de Superficie en México (Temp, Viento, Isobaras y GOES)<br>Observación: {hora_formateada}
    </div>
    <div style="position: fixed; bottom: 20px; left: 10px; background-color: #001f3f; color: white; padding: 8px 15px; border-radius: 5px; z-index: 9999; font-size: 14px; font-weight: bold; font-family: Arial, sans-serif; box-shadow: 2px 2px 5px rgba(0,0,0,0.5);">
        Elaborado por: {FIRMA_ELABORADO_POR}
    </div>
    <div style="position: fixed; top: 75px; left: 10px; background-color: rgba(255,255,255,0.95); padding: 10px; border-radius: 5px; z-index: 9999; font-size: 11px; font-family: Arial, sans-serif; border: 1px solid #ccc; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); color: #333;">
        <b style="font-size: 12px; color: #111;">Reglas de Vuelo (Borde del Círculo)</b><br>
        <span style="color: #28a745; font-weight: bold;">🟢 VFR</span> (Techo >3,000ft / Vis >5sm)<br>
        <span style="color: #007bff; font-weight: bold;">🔵 MVFR</span> (Techo 1,000-3,000ft / Vis 3-5sm)<br>
        <span style="color: #dc3545; font-weight: bold;">🔴 IFR</span> (Techo 500-<1,000ft / Vis 1-<3sm)<br>
        <span style="color: #d63384; font-weight: bold;">🟣 LIFR</span> (Techo <500ft / Vis <1sm)<br>
        <hr style="margin: 6px 0; border: 0; border-top: 1px solid #ccc;">
        <b style="font-size: 12px; color: #111;">Cobertura Nubosa (Relleno)</b><br>
        SKC (⚪) | FEW (1/4 🔵) | SCT (1/2 🔵) | BKN (3/4 🔵) | OVC (🔵)<br>
        <hr style="margin: 6px 0; border: 0; border-top: 1px solid #ccc;">
        <b style="font-size: 12px; color: #111;">Viento (Mástil)</b><br>
        La línea negra indica la dirección <b>DE DONDE</b> viene el viento.
    </div>
"""
mapa_interactivo.get_root().html.add_child(folium.Element(consolidated_overlays))

# --- GENERACIÓN DE ESTACIONES METAR (CON DIRECCIÓN DE VIENTO) ---
capa_estaciones = folium.FeatureGroup(name='Estaciones METAR', show=True)

for idx, row in df_puntos.iterrows():
    icao, lat, lon, obs_time = row['ICAO'], row['lat'], row['lon'], row['obsTime']
    cloud_cover = row.get('cloud_cover', 'SKC')
    borde_vuelo = row.get('border_color', '#28a745') 
    nombre_estacion = nombres_aeropuertos.get(icao, "Desconocida")
    taf_texto = tafs_dict.get(icao, "No hay TAF disponible.")
    
    try: temp = f"{int(round(float(row['temp'])))}" if pd.notnull(row['temp']) else "--"
    except: temp = "--"
    try: dewp = f"{int(round(float(row['dewp'])))}" if pd.notnull(row['dewp']) else "--"
    except: dewp = "--"
    rh = calcular_humedad(row['temp'], row['dewp'])
    try: wspd_kt = int(round(float(row['wspd']))) if pd.notnull(row['wspd']) else 0
    except: wspd_kt = 0
    try: wgst_kt = f"{int(round(float(row['wgst'])))}" if pd.notnull(row['wgst']) else "--"
    except: wgst_kt = "--"
    try: wdir_deg = int(float(row['wdir'])) if pd.notnull(row['wdir']) and str(row['wdir']).upper() != 'VRB' else 0
    except: wdir_deg = 0
    wdir_card = grados_a_cardinal(row['wdir'])
    try: altim = f"{float(row['altim']):.1f}" if pd.notnull(row['altim']) else "--"
    except: altim = "--"

    vis_km = "--"
    if pd.notnull(row['visib']):
        try:
            vis_clean_str = str(row['visib']).replace('+', '').replace('V', '').strip()
            vis_km = f"{int(round(float(vis_clean_str) * 1.609))} km ({str(row['visib'])} sm)"
        except: vis_km = str(row['visib'])

    color_nubes = {
        'OVC': '#0033cc', 'BKN': 'conic-gradient(#0033cc 0% 75%, white 75% 100%)',
        'SCT': 'conic-gradient(#0033cc 0% 50%, white 50% 100%)',
        'FEW': 'conic-gradient(#0033cc 0% 25%, white 25% 100%)', 'SKC': 'white'
    }
    fondo_css = color_nubes.get(cloud_cover, 'white')

    # MAGIA DE VIENTO: Agregamos un mástil ("barba") que rota según los grados exactos del METAR
    icono_y_viento = f"""
    <div style="position: relative; width: 24px; height: 24px; transform: translate(-4px, -4px);">
        <div style="position: absolute; top: 11px; left: 12px; width: 18px; height: 2.5px; background-color: #111; 
                    transform-origin: 0% 50%; transform: rotate({wdir_deg - 90}deg); z-index: 1; 
                    box-shadow: 1px 1px 2px rgba(0,0,0,0.5);"></div>
        <div style="position: absolute; top: 4px; left: 4px; width: 16px; height: 16px; border-radius: 50%; 
                    border: 3.5px solid {borde_vuelo}; background: {fondo_css}; 
                    box-shadow: 1px 1px 4px rgba(0,0,0,0.6); z-index: 2;"></div>
    </div>
    """

    popup_html = f"""
    <div style="width: 340px; font-family: 'Segoe UI', sans-serif; color: #333;">
        <h3 style="margin: 0 0 5px 0; font-size: 16px;">Estación: {nombre_estacion}</h3>
        <div style="font-size: 11px; border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-bottom: 10px; display: flex; justify-content: space-between;">
            <span>Station ID: <b style="color: #c00000;">{icao}</b></span>
            <span>Lat: {lat:.2f} &nbsp;&nbsp; Lon: {lon:.2f}</span>
        </div>
        <div style="text-align: center; font-size: 12px; font-weight: bold; background-color: #f8f9fa; padding: 5px; border-radius: 4px; margin-bottom: 15px;">
            Reporte: {obs_time}
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding: 0 10px;">
            <div style="text-align: center;">
                <div style="font-size: 42px; line-height: 1; font-weight: 300;">{temp}°C</div>
                <div style="font-size: 12px; color: #666; margin-top: 5px; font-weight: bold;">Sensación: {temp}°C</div>
            </div>
            <div style="text-align: center;">
                <div style="width: 50px; height: 50px; border: 2px solid #555; border-radius: 50%; display: flex; align-items: center; justify-content: center; position: relative; margin: 0 auto;">
                    <div style="position: absolute; top: 0; left: 50%; width: 100%; height: 100%; transform: translateX(-50%) rotate({wdir_deg}deg);">
                        <div style="width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 10px solid #222; position: absolute; top: -5px; left: calc(50% - 6px);"></div>
                    </div>
                    <div style="z-index: 10; font-size: 14px; font-weight: bold; line-height: 1.1; background: white; padding: 2px; border-radius: 50%;">
                        {wspd_kt}<br><span style="font-size: 9px; font-weight: normal;">kt</span>
                    </div>
                </div>
                <div style="font-size: 11px; margin-top: 6px; font-weight: bold;">Viento de {wdir_card}</div>
                <div style="font-size: 11px; color: #555;">Ráfagas: {wgst_kt} kt</div>
            </div>
        </div>
        <table style="width: 100%; font-size: 12px; text-align: left; border-top: 1px solid #ddd; padding-top: 10px; margin-bottom: 10px;">
            <tr><td style="padding-bottom: 5px;">Punto de Rocío: <b>{dewp}°C</b></td><td style="padding-bottom: 5px;">Presión: <b>{altim} inHg</b></td></tr>
            <tr><td>Humedad: <b>{rh}%</b></td><td>Visibilidad: <b>{vis_km}</b></td></tr>
            <tr><td colspan="2" style="padding-top: 5px; border-top: 1px dashed #eee;">Cielo: <b>{cloud_cover}</b></td></tr>
        </table>
        <details style="background-color: #f1f8ff; border: 1px solid #cce5ff; border-radius: 5px; padding: 5px;">
            <summary style="font-size: 12px; font-weight: bold; color: #004085; cursor: pointer;">📋 Ver Pronóstico TAF</summary>
            <div style="margin-top: 8px; padding: 8px; background-color: #ffffff; border: 1px solid #ddd; border-radius: 3px; font-family: monospace; font-size: 10px; color: #333; max-height: 120px; overflow-y: auto; white-space: pre-wrap;">{taf_texto}</div>
        </details>
    </div>
    """
    
    folium.Marker(
        location=[lat, lon], popup=folium.Popup(popup_html, max_width=380),
        tooltip=f"{nombre_estacion} ({icao})",
        icon=folium.DivIcon(html=icono_y_viento, icon_size=(24, 24), icon_anchor=(12, 12))
    ).add_to(capa_estaciones)

capa_estaciones.add_to(mapa_interactivo)
folium.LayerControl().add_to(mapa_interactivo)

# --- BOTÓN DE REFRESCAR Y RENDERIZADO ---
col1, col2 = st.columns([4, 1])
with col1:
    st.success(f"Datos actualizados en la web. Última observación: **{hora_formateada}**")
with col2:
    if st.button("🔄 Refrescar Datos"):
        st.cache_data.clear()
        st.rerun()

st_folium(mapa_interactivo, width=1300, height=700, returned_objects=[])
