"use strict";
const get = id => document.getElementById(id);
const form = get("trip-form"), select = get("trip"), run = get("run");
const gpsColor = "#275dae", routeColor = "#006253", classifierColor = "#a0470c";
const routeView = get("route-view");
let data, street, streetPromise;
const timed = new Intl.NumberFormat("en", {maximumFractionDigits: 1});
const percentage = new Intl.NumberFormat("en", {style: "percent", maximumFractionDigits: 2});

function connection(state, text) {
  const el = get("connection"); el.dataset.state = state;
  el.replaceChildren(document.createElement("i"), text);
}
function error(text) {
  get("error").textContent = text || "";
  get("error").hidden = !text;
}
async function request(path, method = "GET") {
  const res = await fetch(path, {method, headers: {Accept: "application/json"}});
  if (!res.ok) {
    const payload = await res.json().catch(() => ({}));
    throw new Error(payload.detail || "Request failed (" + res.status + ")");
  }
  return res.json();
}
async function init() {
  try {
    const trips = await request("/api/trips");
    select.replaceChildren(...trips.map(id => new Option(id, id)));
    if (!trips.length) throw new Error("No trips are available.");
    select.disabled = run.disabled = false; connection("ready", "API connected");
  } catch (e) {
    connection("error", "API unavailable"); error(e.message);
    select.replaceChildren(new Option("No trips loaded"));
  }
}
function visibleRoutes() {
  if (!data) return [];
  const routes = [];
  if (routeView.value !== "classifier") {
    routes.push({segments: data.predicted_route, color: routeColor, dashed: false});
  }
  if (routeView.value !== "viterbi") {
    routes.push({segments: data.classifier_route, color: classifierColor, dashed: true});
  }
  return routes;
}

function syncRouteLegend() {
  const show = get("route-layer").checked;
  get("viterbi-key").hidden = !data || !show || routeView.value === "classifier";
  get("classifier-key").hidden = !data || !show || routeView.value === "viterbi";
  get("top1_acc_hmm").closest("div").classList.toggle("primary", routeView.value !== "classifier");
  get("top1_acc_clf").closest("div").classList.toggle("primary", routeView.value === "classifier");
}

function useResult(result) {
  data = result;
  const comparison = Array.isArray(data.classifier_route);
  routeView.disabled = false;
  routeView.options[1].disabled = routeView.options[2].disabled = !comparison;
  if (!comparison) routeView.value = "viterbi";
  get("map-subtitle").textContent = data.trip_id + " · " + data.n_points.toLocaleString() + " observations";
  const metrics = data.metrics;
  for (const key of ["top1_acc_hmm", "top1_acc_clf", "route_continuity", "candidate_miss_rate"]) {
    get(key).textContent = metrics[key] == null ? "—" : percentage.format(metrics[key]);
  }
  get("inference_ms").textContent = timed.format(metrics.inference_ms) + " ms";
  get("metric-status").textContent = metrics.top1_acc_hmm == null ? "Reference metrics unavailable" : "Calculated on the API server";
  syncRouteLegend();
  showRouteOnMap();
}
form.addEventListener("submit", async event => {
  event.preventDefault(); error();
  const id = select.value;
  run.disabled = select.disabled = true; run.textContent = "Reconstructing…";
  get("metric-status").textContent = "Running inference…";
  try {
    useResult(await request("/api/predict/" + encodeURIComponent(id) + "?comparison=1", "POST"));
    connection("ready", "API connected");
  } catch (e) {
    get("metric-status").textContent = "Prediction failed"; error(e.message);
  } finally {
    select.disabled = run.disabled = false; run.textContent = "Reconstruct route →";
  }
});
function loadLeaflet() {
  const base = "https://unpkg.com/leaflet@1.9.4/dist/";
  const css = document.createElement("link");
  const script = document.createElement("script");
  css.rel = "stylesheet";
  css.href = base + "leaflet.css";
  css.integrity = "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=";
  script.src = base + "leaflet.js";
  script.integrity = "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=";
  const load = element => new Promise((resolve, reject) => {
    element.crossOrigin = "anonymous";
    element.onload = resolve;
    element.onerror = reject;
    document.head.append(element);
  });
  return Promise.all([load(css), load(script)]).catch(error => {
    css.remove();
    script.remove();
    throw error;
  });
}

function openStreetMap() {
  if (streetPromise) return streetPromise;
  const status = get("map-status");
  status.textContent = "Loading street map...";
  streetPromise = loadLeaflet().then(() => {
    const L = window.L;
    const map = L.map("street-map", {zoomControl: false, maxZoom: 19});
    map.setView([0, 0], 2);
    map.createPane("gps-points").style.zIndex = 390;
    street = {map, route: L.layerGroup(), gps: L.layerGroup(),
              gpsRenderer: L.canvas({pane: "gps-points"}), bounds: null};
    L.control.scale({imperial: false}).addTo(map);
    let loaded = 0;
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
    }).on("loading", () => { loaded = 0; })
      .on("tileload", () => {
        loaded += 1;
        status.textContent = "";
      })
      .on("load", () => {
        if (!loaded) status.textContent = "Street tiles unavailable. Trip overlays remain visible.";
      }).addTo(map);
    new ResizeObserver(() => map.invalidateSize()).observe(get("street-map"));
    return street;
  }).catch(error => {
    streetPromise = null;
    status.textContent = "Street map could not load. Check your internet connection.";
    throw error;
  });
  return streetPromise;
}

function syncStreetLayers() {
  if (!street) return;
  for (const [layer, toggle] of [[street.route, "route-layer"], [street.gps, "gps-layer"]]) {
    if (get(toggle).checked) layer.addTo(street.map);
    else layer.remove();
  }
}

function drawStreetRoutes() {
  if (!street) return;
  street.route.clearLayers();
  const L = window.L;
  for (const {segments, color, dashed} of visibleRoutes()) {
    if (!segments.length) continue;
    const lines = segments.map(({coords}) => coords);
    const dashArray = dashed ? "8 7" : null;
    L.polyline(lines, {color: "#fff", weight: 8, opacity: .95,
                      dashArray, interactive: false}).addTo(street.route);
    L.polyline(lines, {color, weight: 4.5, dashArray, interactive: false}).addTo(street.route);
  }
}

function fitStreet() {
  if (!street?.bounds?.isValid()) return;
  street.map.invalidateSize();
  street.map.fitBounds(street.bounds, {padding: [30, 30], maxZoom: 17, animate: false});
}

async function showRouteOnMap() {
  const result = data;
  try {
    await openStreetMap();
  } catch {
    return;
  }
  if (result !== data) return;

  const L = window.L;
  const segments = [...result.predicted_route, ...(result.classifier_route ?? [])];
  street.bounds = L.latLngBounds([...result.gps, ...segments.flatMap(({coords}) => coords)]);
  const hasGeometry = street.bounds.isValid();
  get("placeholder").hidden = hasGeometry;
  get("placeholder").textContent = "No route geometry for this trip.";
  if (hasGeometry) {
    const center = street.bounds.getCenter();
    get("bounds").textContent = center.lat.toFixed(4) + "°, " + center.lng.toFixed(4) + "°";
    fitStreet();
  } else {
    get("bounds").textContent = "";
  }
  street.gps.clearLayers();
  for (const point of result.gps) {
    L.circleMarker(point, {radius: 2.5, stroke: false, fillColor: gpsColor,
                          fillOpacity: .7, interactive: false,
                          renderer: street.gpsRenderer}).addTo(street.gps);
  }
  drawStreetRoutes();
  syncStreetLayers();
}

for (const key of ["gps-layer", "route-layer"]) {
  get(key).addEventListener("change", () => { syncStreetLayers(); syncRouteLegend(); });
}
routeView.addEventListener("change", () => {
  syncRouteLegend();
  drawStreetRoutes();
});
get("reset").addEventListener("click", fitStreet);
get("zoom-in").addEventListener("click", () => street?.map.zoomIn());
get("zoom-out").addEventListener("click", () => street?.map.zoomOut());

openStreetMap().catch(() => {});
init();
