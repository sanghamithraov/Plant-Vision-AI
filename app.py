import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np
import time
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Plant Vision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# GLOBAL DESIGN
# =========================================================

st.markdown("""
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap'
);

:root {
    --green: #8cffc0;
    --green-bright: #39f5a0;
    --muted: #8fa99b;
    --panel: rgba(10, 26, 20, 0.72);
    --border: rgba(140, 255, 192, 0.16);
}

.stApp {
    background:
        radial-gradient(
            ellipse at 50% 0%,
            rgba(25, 112, 75, 0.22),
            transparent 46%
        ),
        radial-gradient(
            ellipse at 90% 60%,
            rgba(18, 93, 68, 0.12),
            transparent 40%
        ),
        #040a08;

    color: #edf9f1;
    font-family: 'Manrope', -apple-system, "Segoe UI", Roboto, sans-serif;
}

.block-container {
    max-width: 1360px;
    padding: 1.2rem 2rem 4rem;
}

header {
    background: transparent !important;
}

#MainMenu,
footer {
    visibility: hidden;
}

/* Streamlit uploader, restyled as a scan-bay */

[data-testid="stFileUploader"] {
    background: rgba(8, 22, 17, 0.6);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 18px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
    border: 1.5px dashed rgba(140, 255, 192, 0.35);
    border-radius: 16px;
}

[data-testid="stFileUploader"] button {
    background: rgba(57, 245, 160, 0.14);
    border: 1px solid rgba(140, 255, 192, 0.35);
    color: var(--green);
    border-radius: 10px;
}

[data-testid="stImage"] img {
    border-radius: 18px;
}

[data-testid="stProgressBar"] {
    background: rgba(255,255,255,0.06);
    border-radius: 20px;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #1abf82, #8cffc0);
    border-radius: 20px;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1rem 1rem 3rem;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# REAL 3D HERO (Three.js WebGL leaf, not CSS/SVG fake-3D)
# =========================================================

hero_html = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>

* { box-sizing: border-box; }

html, body {
    margin: 0;
    padding: 0;
    background: transparent;
    color: #edf9f1;
    font-family: 'Manrope', -apple-system, "Segoe UI", Roboto, sans-serif;
    overflow: hidden;
}

.scene {
    position: relative;
    width: 100%;
    height: 560px;
    border-radius: 30px;
    overflow: hidden;
    background:
        radial-gradient(ellipse at 50% 40%, rgba(28,110,73,0.32), transparent 55%),
        linear-gradient(150deg, rgba(11,34,24,0.75), rgba(3,8,6,0.98));
    border: 1px solid rgba(140,255,192,0.14);
}

canvas#leafCanvas {
    position: absolute;
    inset: 0;
    width: 100% !important;
    height: 100% !important;
}

/* Grain / vignette for a cinematic feel */

.vignette {
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: radial-gradient(
        ellipse at 50% 50%,
        transparent 40%,
        rgba(2,6,4,0.55) 100%
    );
}

.hero-content {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 34px;
    text-align: center;
    padding: 0 16px;
    z-index: 5;
    pointer-events: none;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    color: #8cffc0;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 9px 15px;
    border: 1px solid rgba(140,255,192,0.2);
    border-radius: 50px;
    background: rgba(5,20,13,0.7);
}

.status-dot {
    width: 6px;
    height: 6px;
    background: #39f5a0;
    border-radius: 50%;
    box-shadow: 0 0 12px #39f5a0;
    display: inline-block;
}

h1 {
    margin: 14px 0 8px;
    font-size: clamp(38px, 6.5vw, 66px);
    font-weight: 800;
    letter-spacing: -2.5px;
    line-height: 1.03;
    background: linear-gradient(100deg, #ffffff, #c5ffe0, #62f6ae);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.subtitle {
    color: #a1b9aa;
    font-size: 13.5px;
    line-height: 1.8;
    max-width: 460px;
    margin: 0 auto;
}

.corner {
    position: absolute;
    top: 24px;
    left: 26px;
    color: rgba(180,220,195,0.55);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    z-index: 5;
}

.corner.right {
    left: auto;
    right: 26px;
    text-align: right;
}

.hint {
    position: absolute;
    bottom: 18px;
    right: 26px;
    color: rgba(180,220,195,0.4);
    font-family: 'DM Mono', monospace;
    font-size: 9.5px;
    letter-spacing: 1px;
    z-index: 5;
}

@media (max-width: 500px) {
    .scene { height: 460px; }
    .hero-content { bottom: 22px; }
    .corner { top: 14px; left: 14px; }
    .corner.right { right: 14px; }
    .hint { display: none; }
}

@media (prefers-reduced-motion: reduce) {
    canvas#leafCanvas { display: none; }
}

</style>
</head>
<body>

<div class="scene">

    <canvas id="leafCanvas"></canvas>
    <div class="vignette"></div>

    <div class="corner">PV / 001<br>BOTANICAL INTELLIGENCE</div>
    <div class="corner right">SYSTEM ONLINE<br><span style="color:#8cffc0">●</span> MODEL READY</div>

    <div class="hero-content">
        <div class="eyebrow"><span class="status-dot"></span>AI-POWERED PLANT HEALTH</div>
        <h1>Plant Vision</h1>
        <p class="subtitle">Nature, decoded.<br>Reveal the hidden health of every leaf.</p>
    </div>

</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/CopyShader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/LuminosityHighPassShader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/EffectComposer.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/RenderPass.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/ShaderPass.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/UnrealBloomPass.js"></script>
<script>

const canvas = document.getElementById('leafCanvas');
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(42, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
camera.position.set(0, 0.35, 6.4);

const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight);

// Physically-based lighting response instead of flat shading --
// this alone is most of what makes a scene read as "real" vs "toy"
renderer.physicallyCorrectLights = true;
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;

// ---------- Procedural sky/environment sphere (for reflections) ----------
// No external HDRI needed -- a gradient shader sphere gives the
// PBR material something believable to reflect

const envScene = new THREE.Scene();

const skyGeo = new THREE.SphereGeometry(50, 32, 32);
const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    uniforms: {
        topColor: { value: new THREE.Color(0x11402c) },
        bottomColor: { value: new THREE.Color(0x020705) },
        glow: { value: new THREE.Color(0x39f5a0) }
    },
    vertexShader: `
        varying vec3 vPos;
        void main() {
            vPos = position;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        varying vec3 vPos;
        uniform vec3 topColor;
        uniform vec3 bottomColor;
        uniform vec3 glow;
        void main() {
            float h = normalize(vPos).y * 0.5 + 0.5;
            vec3 base = mix(bottomColor, topColor, h);
            float rim = pow(1.0 - abs(normalize(vPos).y), 4.0);
            gl_FragColor = vec4(base + glow * rim * 0.25, 1.0);
        }
    `
});

envScene.add(new THREE.Mesh(skyGeo, skyMat));
scene.background = null;

const cubeRenderTarget = new THREE.WebGLCubeRenderTarget(256, {
    format: THREE.RGBFormat,
    generateMipmaps: true,
    minFilter: THREE.LinearMipmapLinearFilter
});

const cubeCamera = new THREE.CubeCamera(0.1, 50, cubeRenderTarget);

// ---------- Lighting ----------

scene.add(new THREE.HemisphereLight(0x9dffd4, 0x03110a, 1.0));

const key = new THREE.PointLight(0x8cffc0, 6, 20, 2);
key.position.set(3, 3, 4);
scene.add(key);

const rim = new THREE.PointLight(0x39f5a0, 4, 20, 2);
rim.position.set(-4, -1, -3);
scene.add(rim);

const fill = new THREE.DirectionalLight(0xd8fff0, 0.6);
fill.position.set(0, 5, 2);
scene.add(fill);

// ---------- Procedural leaf texture: color + veins + natural mottling ----------
// A believable leaf is never flat-colored -- real leaves have patchy
// darker/lighter green zones, slightly warm-toned veins, and subtle
// surface blemishes. This is drawn on a canvas at runtime.

function buildLeafColorTexture() {
    const size = 1024;
    const c = document.createElement('canvas');
    c.width = size; c.height = size;
    const ctx = c.getContext('2d');

    // base gradient: darker at edges, richer green in the middle
    const grad = ctx.createRadialGradient(size/2, size*0.42, size*0.08, size/2, size/2, size*0.65);
    grad.addColorStop(0, '#3fae6e');
    grad.addColorStop(0.5, '#237c4e');
    grad.addColorStop(1, '#0f4a2e');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);

    // patchy mottling -- irregular soft blotches of slightly
    // different green tones, like real chlorophyll variation
    for (let i = 0; i < 60; i++) {
        const x = Math.random() * size;
        const y = Math.random() * size;
        const r = 30 + Math.random() * 90;
        const tone = Math.random() > 0.5 ? 'rgba(90,190,120,0.10)' : 'rgba(15,60,35,0.14)';
        const blot = ctx.createRadialGradient(x, y, 0, x, y, r);
        blot.addColorStop(0, tone);
        blot.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = blot;
        ctx.fillRect(x - r, y - r, r * 2, r * 2);
    }

    // fine grain
    for (let i = 0; i < 12000; i++) {
        const v = Math.random() * 30 - 15;
        ctx.fillStyle = `rgba(${v > 0 ? 255 : 0},${v > 0 ? 255 : 0},${v > 0 ? 255 : 0},${Math.abs(v) / 200})`;
        ctx.fillRect(Math.random() * size, Math.random() * size, 1.3, 1.3);
    }

    // veins -- warm pale-gold, not pure white, and slightly irregular
    ctx.strokeStyle = 'rgba(214, 232, 170, 0.55)';
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(size / 2, size * 0.05);
    ctx.bezierCurveTo(size * 0.49, size * 0.4, size * 0.51, size * 0.6, size / 2, size * 0.97);
    ctx.stroke();

    ctx.lineWidth = 2.6;
    ctx.strokeStyle = 'rgba(214, 232, 170, 0.4)';
    for (let i = 1; i <= 7; i++) {
        const y = size * (0.1 + i * 0.105);
        const spread = size * 0.36 * (i / 7);
        const wobble = (Math.random() - 0.5) * 20;
        ctx.beginPath();
        ctx.moveTo(size / 2, y);
        ctx.quadraticCurveTo(size / 2 - spread * 0.5, y - spread * 0.35 + wobble, size / 2 - spread, y - spread * 0.55);
        ctx.moveTo(size / 2, y);
        ctx.quadraticCurveTo(size / 2 + spread * 0.5, y - spread * 0.35 - wobble, size / 2 + spread, y - spread * 0.55);
        ctx.stroke();
    }

    const tex = new THREE.CanvasTexture(c);
    tex.encoding = THREE.sRGBEncoding;
    return tex;
}

const leafColorTexture = buildLeafColorTexture();

// ---------- Blurred plantation-style backdrop ----------
// Soft green bokeh shapes standing in for an out-of-focus garden/
// plantation background -- no external photo needed, so no
// copyright concerns, but reads as "shot in a leafy environment"

function buildPlantationBackdrop() {
    const size = 1024;
    const c = document.createElement('canvas');
    c.width = size; c.height = size;
    const ctx = c.getContext('2d');

    const skyGrad = ctx.createLinearGradient(0, 0, 0, size);
    skyGrad.addColorStop(0, '#0c2418');
    skyGrad.addColorStop(1, '#02100a');
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, size, size);

    // soft out-of-focus leaf blobs at varying depth (size/opacity)
    for (let i = 0; i < 55; i++) {
        const x = Math.random() * size;
        const y = size * 0.25 + Math.random() * size * 0.75;
        const r = 40 + Math.random() * 160;
        const hue = 100 + Math.random() * 40;
        const light = 18 + Math.random() * 22;
        const blob = ctx.createRadialGradient(x, y, 0, x, y, r);
        blob.addColorStop(0, `hsla(${hue}, 55%, ${light}%, 0.5)`);
        blob.addColorStop(1, 'hsla(0,0%,0%,0)');
        ctx.fillStyle = blob;
        ctx.fillRect(x - r, y - r, r * 2, r * 2);
    }

    // warm sunlight bokeh dots filtering through canopy
    for (let i = 0; i < 18; i++) {
        const x = Math.random() * size;
        const y = Math.random() * size * 0.7;
        const r = 8 + Math.random() * 22;
        const dot = ctx.createRadialGradient(x, y, 0, x, y, r);
        dot.addColorStop(0, 'rgba(255,244,200,0.35)');
        dot.addColorStop(1, 'rgba(255,244,200,0)');
        ctx.fillStyle = dot;
        ctx.fillRect(x - r, y - r, r * 2, r * 2);
    }

    const tex = new THREE.CanvasTexture(c);
    tex.encoding = THREE.sRGBEncoding;
    return tex;
}

const backdropTexture = buildPlantationBackdrop();
scene.background = backdropTexture;

// ---------- Real 3D leaf geometry ----------

function buildLeafShape() {
    const shape = new THREE.Shape();
    shape.moveTo(0, -2.1);
    shape.bezierCurveTo(-1.9, -1.15, -2.05, 0.75, -1.7, 1.6);
    shape.bezierCurveTo(-1.35, 2.5, -0.4, 3.05, 0, 3.15);
    shape.bezierCurveTo(0.4, 3.05, 1.35, 2.5, 1.7, 1.6);
    shape.bezierCurveTo(2.05, 0.75, 1.9, -1.15, 0, -2.1);
    return shape;
}

const extrudeSettings = {
    steps: 2,
    depth: 0.16,
    bevelEnabled: true,
    bevelThickness: 0.05,
    bevelSize: 0.045,
    bevelSegments: 6
};

const leafGeometry = new THREE.ExtrudeGeometry(buildLeafShape(), extrudeSettings);
leafGeometry.center();
leafGeometry.computeVertexNormals();

// A believable leaf material: waxy clearcoat on the surface,
// slight light transmission at the edges (how real leaves glow
// when backlit), and the procedural color+vein texture above
const leafMaterial = new THREE.MeshPhysicalMaterial({
    map: leafColorTexture,
    roughness: 0.34,
    metalness: 0.0,
    clearcoat: 0.7,
    clearcoatRoughness: 0.24,
    reflectivity: 0.35,
    transmission: 0.2,
    thickness: 0.35,
    ior: 1.35,
    bumpMap: leafColorTexture,
    bumpScale: 0.02,
    envMapIntensity: 1.2
});

const leafGroup = new THREE.Group();
const leafMesh = new THREE.Mesh(leafGeometry, leafMaterial);
leafGroup.add(leafMesh);

scene.add(leafGroup);

// ---------- Water-droplet sparkle particles (additive glow) ----------

const particleCount = 90;
const particlePositions = new Float32Array(particleCount * 3);

for (let i = 0; i < particleCount; i++) {
    particlePositions[i * 3] = (Math.random() - 0.5) * 12;
    particlePositions[i * 3 + 1] = (Math.random() - 0.5) * 9;
    particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 8 - 1;
}

const particleGeo = new THREE.BufferGeometry();
particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));

const particleMat = new THREE.PointsMaterial({
    color: 0xbfffdb,
    size: 0.05,
    transparent: true,
    opacity: 0.85,
    blending: THREE.AdditiveBlending,
    depthWrite: false
});

const particles = new THREE.Points(particleGeo, particleMat);
scene.add(particles);

// ---------- Post-processing: bloom for real light glow ----------

const composer = new THREE.EffectComposer(renderer);
composer.addPass(new THREE.RenderPass(scene, camera));

const bloomPass = new THREE.UnrealBloomPass(
    new THREE.Vector2(canvas.clientWidth, canvas.clientHeight),
    0.85,   // strength
    0.55,   // radius
    0.22    // threshold
);
composer.addPass(bloomPass);

function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    bloomPass.setSize(w, h);
}

window.addEventListener('resize', resize);
resize();

let t = 0;
let envFrame = 0;

function animate() {
    requestAnimationFrame(animate);
    t += 0.01;

    // continuous turntable rotation -- no drag needed
    leafGroup.rotation.y = t * 0.35;
    leafGroup.rotation.x = Math.sin(t * 0.3) * 0.12;
    leafGroup.position.y = Math.sin(t) * 0.14;

    particles.rotation.y += 0.0006;

    key.position.x = Math.sin(t * 0.5) * 4;
    key.position.z = Math.cos(t * 0.5) * 4 + 1;
    rim.position.x = Math.cos(t * 0.4) * -4;

    // refresh the reflection every few frames (cheap enough at this
    // resolution, avoids paying the cost every single frame)
    envFrame++;
    if (envFrame % 4 === 0) {
        leafMesh.visible = false;
        cubeCamera.position.copy(leafGroup.position);
        cubeCamera.update(renderer, envScene);
        leafMaterial.envMap = cubeRenderTarget.texture;
        leafMesh.visible = true;
    }

    composer.render();
}

animate();

</script>
</body>
</html>
"""

components.html(hero_html, height=560, scrolling=False)


# =========================================================
# SECTION HEADING
# =========================================================

st.markdown("""
<div style="margin:38px 0 18px; display:flex; justify-content:space-between;
    align-items:center; flex-wrap:wrap; gap:10px;">
<div>
<div style="color:#8cffc0; font-family:'DM Mono',monospace; font-size:11px;
    letter-spacing:2px; margin-bottom:7px;">/ 01 — SCAN BAY</div>
<div style="font-size:28px; font-weight:800; letter-spacing:-1px;">Scan your leaf</div>
</div>
<div style="color:#8fa99b; font-size:12px; border:1px solid rgba(140,255,192,.16);
    padding:9px 13px; border-radius:30px;">MobileNetV2 · Image classification</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")


model = load_model()


# =========================================================
# GET CLASS NAMES
# =========================================================

# GET CLASS NAMES
with open("class_names.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

# =========================================================
# CHECK MODEL OUTPUT
# =========================================================

model_output_classes = model.output_shape[-1]

if model_output_classes != len(class_names):
    st.error(
        "The number of model output classes does not match the number "
        "of dataset folders. Please check the model and class names."
    )
    st.stop()


# =========================================================
# UPLOAD SECTION (scan-bay styled panel)
# =========================================================

left, right = st.columns([1, 1], gap="large")

with left:

    st.markdown("""
<div style="background:var(--panel); border:1px solid rgba(140,255,192,.16);
    border-radius:22px; padding:24px; margin-bottom:12px;">
<div style="color:#8cffc0; font-family:'DM Mono',monospace; font-size:11px;
    letter-spacing:2px;">IMAGE INPUT</div>
<div style="font-size:23px; font-weight:800; margin:12px 0 8px;">Load a specimen</div>
<div style="color:#8fa99b; font-size:13px; line-height:1.8;">
Drop a clear photo of a plant leaf below. For best accuracy, keep the
leaf centered, in focus, and free of overlapping foliage.
</div>
</div>
""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )


with right:

    st.markdown("""
<div style="background:var(--panel); border:1px solid rgba(140,255,192,.16);
    border-radius:22px; padding:24px; min-height:150px;">
<div style="color:#8cffc0; font-family:'DM Mono',monospace; font-size:11px;
    letter-spacing:2px;">HOW IT WORKS</div>
<div style="font-size:23px; font-weight:800; margin:12px 0 8px;">From leaf to insight</div>
<div style="color:#8fa99b; font-size:13px; line-height:1.8;">
Your image is resized to 224 × 224 pixels, scanned by a fine-tuned
MobileNetV2 network, and ranked across 38 possible plant conditions.
</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SCANNING ANIMATION (shown while the model is predicting)
# =========================================================

scanning_html = """
<div style="
    position:relative;
    height:170px;
    border-radius:20px;
    overflow:hidden;
    border:1px solid rgba(140,255,192,.25);
    background:
        repeating-linear-gradient(
            0deg,
            rgba(140,255,192,0.05) 0px,
            rgba(140,255,192,0.05) 1px,
            transparent 1px,
            transparent 26px
        ),
        radial-gradient(ellipse at 50% 50%, rgba(28,110,73,0.28), #060f0b 75%);
    font-family:'DM Mono', monospace;
    color:#8cffc0;
    display:flex;
    align-items:center;
    justify-content:center;
    letter-spacing:2px;
    font-size:12px;
">
    <div style="
        position:absolute;
        left:0; right:0;
        height:3px;
        background:linear-gradient(90deg, transparent, #39f5a0, transparent);
        box-shadow:0 0 18px #39f5a0;
        animation: scanline 1.6s ease-in-out infinite;
    "></div>
    ANALYZING SPECIMEN...
</div>

<style>
@keyframes scanline {
    0%   { top: 6%; opacity: 0.2; }
    50%  { top: 90%; opacity: 1; }
    100% { top: 6%; opacity: 0.2; }
}
</style>
"""


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # PREPROCESSING
    # -----------------------------------------------------

    image = tf.keras.utils.load_img(uploaded_file, target_size=(224, 224))
    image_array = tf.keras.utils.img_to_array(image)

    # IMPORTANT:
    # Do not divide by 255 -- the model already contains
    # Rescaling(1./127.5, offset=-1) as its first layer.

    image_array = np.expand_dims(image_array, axis=0)

    # -----------------------------------------------------
    # SHOW SCANNING ANIMATION, THEN PREDICT
    # -----------------------------------------------------

    scan_placeholder = st.empty()
    scan_placeholder.markdown(scanning_html, unsafe_allow_html=True)

    predictions = model.predict(image_array, verbose=0)[0]

    # small deliberate pause so the scan animation is visible
    # even when prediction itself is very fast
    time.sleep(0.5)
    scan_placeholder.empty()

    # -----------------------------------------------------
    # TOP 3 PREDICTIONS
    # -----------------------------------------------------

    top_3_indices = np.argsort(predictions)[-min(3, len(predictions)):][::-1]
    predicted_index = int(top_3_indices[0])
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index] * 100)
    readable_name = predicted_class.replace("___", " - ").replace("_", " ")

    # -----------------------------------------------------
    # RESULT HEADER
    # -----------------------------------------------------

    st.markdown("""
<div style="margin-top:40px; margin-bottom:18px;">
<div style="color:#8cffc0; font-family:'DM Mono',monospace; font-size:11px;
    letter-spacing:2px; margin-bottom:7px;">/ 02 — ANALYSIS</div>
<div style="font-size:28px; font-weight:800; letter-spacing:-1px;">Your results</div>
</div>
""", unsafe_allow_html=True)

    result_left, result_right = st.columns([1, 1], gap="large")

    with result_left:

        st.markdown("""
<div style="background:var(--panel); border:1px solid rgba(140,255,192,.16);
    border-radius:20px; padding:16px;">
""", unsafe_allow_html=True)

        st.image(uploaded_file, caption="Uploaded specimen", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with result_right:

        # ---------- Circular confidence gauge (real SVG, driven by the actual value) ----------

        radius = 54
        circumference = 2 * 3.14159265 * radius
        offset = circumference * (1 - confidence / 100)

        gauge_html = f"""
<div style="background:linear-gradient(145deg, rgba(57,245,160,0.10), rgba(10,26,20,0.85));
    border:1px solid rgba(140,255,192,.24); border-radius:20px; padding:24px;
    height:100%; display:flex; flex-direction:column; align-items:center; text-align:center;">

<div style="color:#8cffc0; font-family:'DM Mono',monospace; font-size:11px;
    letter-spacing:2px; text-transform:uppercase; align-self:flex-start;">
    Analysis complete
</div>

<svg width="150" height="150" viewBox="0 0 140 140" style="margin:14px 0;">
    <circle cx="70" cy="70" r="{radius}" fill="none"
        stroke="rgba(255,255,255,0.08)" stroke-width="10"/>
    <circle cx="70" cy="70" r="{radius}" fill="none"
        stroke="url(#gaugeGradient)" stroke-width="10"
        stroke-linecap="round"
        stroke-dasharray="{circumference:.2f}"
        stroke-dashoffset="{circumference:.2f}"
        transform="rotate(-90 70 70)"
        style="animation: fillGauge 1.2s ease-out forwards;">
    </circle>
    <defs>
        <linearGradient id="gaugeGradient" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#1abf82"/>
            <stop offset="100%" stop-color="#8cffc0"/>
        </linearGradient>
    </defs>
    <text x="70" y="66" text-anchor="middle" font-size="26" font-weight="800"
        fill="#edf9f1" font-family="Manrope, sans-serif">{confidence:.0f}%</text>
    <text x="70" y="86" text-anchor="middle" font-size="9" letter-spacing="1"
        fill="#8fa99b" font-family="'DM Mono', monospace">CONFIDENCE</text>
</svg>

<style>
@keyframes fillGauge {{
    to {{ stroke-dashoffset: {offset:.2f}; }}
}}
</style>

<div style="font-size:24px; font-weight:800; margin-top:4px;">{readable_name}</div>

</div>
"""

        components.html(gauge_html, height=290)

    # -----------------------------------------------------
    # TOP 3 PREDICTIONS — ranked visual list
    # -----------------------------------------------------

    st.markdown("""
<div style="margin-top:34px; margin-bottom:14px; font-size:20px; font-weight:800;">
    Ranked predictions
</div>
""", unsafe_allow_html=True)

    medal_colors = ["#39f5a0", "#8cffc0", "#5a8f74"]

    for rank, index in enumerate(top_3_indices, start=1):

        name = class_names[index].replace("___", " - ").replace("_", " ")
        score = predictions[index] * 100
        color = medal_colors[rank - 1] if rank <= 3 else "#5a8f74"

        st.markdown(f"""
<div style="display:flex; align-items:center; gap:14px; margin-bottom:10px;">
<div style="width:26px; height:26px; border-radius:50%; background:rgba(140,255,192,.12);
    border:1px solid rgba(140,255,192,.3); display:flex; align-items:center;
    justify-content:center; font-family:'DM Mono',monospace; font-size:12px; color:{color};">
    {rank}
</div>
<div style="flex:1;">
<div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:4px;">
    <span style="font-weight:700;">{name}</span>
    <span style="color:{color}; font-family:'DM Mono',monospace;">{score:.2f}%</span>
</div>
<div style="height:8px; background:rgba(255,255,255,0.06); border-radius:10px; overflow:hidden;">
    <div style="height:100%; width:{score:.2f}%; background:linear-gradient(90deg,#1abf82,{color});
        border-radius:10px;"></div>
</div>
</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div style="text-align:center; color:#5f7568; margin-top:60px; font-size:12px;
    font-family:'DM Mono',monospace; letter-spacing:1px;">
Built with TensorFlow · MobileNetV2 · Three.js · Streamlit
</div>
""", unsafe_allow_html=True)