# City Transport Efficiency Dashboard – Expert Edition

## Client
**City Transport Department** – responsible for traffic flow, road infrastructure, and issue resolution.

## How to run
(standard instructions as before)

## Expert enhancements

### 1. Priority Index map metric
- Combines `total_reports` × `unresolved_rate` to highlight districts under **combined pressure**.
- Default map view uses this metric – most actionable for resource allocation.

### 2. Custom CSS styling
- KPI cards with shadow, rounded corners, professional colour.
- Section headers for clear visual hierarchy.

### 3. Enhanced tooltips on map
- Shows district name, total reports, unresolved %, median days, and priority index.
- Helps supervisors triage at a glance.

### 4. Dynamic insights
- Three insights: highest priority district, fastest growing issue, slowest resolution.
- Updates automatically when filters change.

### 5. Export options
- Download full cleaned dataset or filtered view – useful for external reporting.

## Design justifications (for Transport Department)
- **Blue colour scale** on default charts: calm, efficiency‑focused.
- **Warm scale (Reds) for priority index** to signal urgency where needed.
- **Folium background map** gives real‑world context (streets, landmarks).
- **Horizontal bar chart** for issue types because labels are long.
- **Scatter plot** identifies districts that are both high‑volume and high‑unresolved.
- **Priority index** is a custom composite metric that aligns with the client’s goal: “where to send resources first”.

## AI tools used
- ChatGPT for code, debugging, and design rationale.
- Copilot for minor snippets.

## What worked well
- GeoPandas spatial join after fixing property name (`zone_id`).
- Folium choropleth with OpenStreetMap background.
- Caching made filters instant.

## What didn’t
- Timezone issues with datetime subtraction – resolved by stripping tz.
- GeoJSON property naming inconsistency – solved by auto‑detection and manual mapping.