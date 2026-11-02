/**
 * tinyISB Documentation Web App Engine
 * Client-side routing, CMSIS-style view generation, instant search, and theme switcher.
 */

document.addEventListener("DOMContentLoaded", () => {
  const contentArea = document.getElementById("content-area");
  const searchInput = document.getElementById("search-input");
  const themeToggleBtn = document.getElementById("theme-toggle-btn");
  const mobileMenuBtn = document.getElementById("mobile-menu-btn");
  const sidebar = document.getElementById("sidebar");

  // 1. Theme Switcher (Dark / Light)
  const currentTheme = localStorage.getItem("tinyisb-theme") || 
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", currentTheme);
  updateThemeIcon(currentTheme);

  themeToggleBtn.addEventListener("click", () => {
    const theme = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("tinyisb-theme", theme);
    updateThemeIcon(theme);
  });

  function updateThemeIcon(theme) {
    themeToggleBtn.textContent = theme === "dark" ? "☀️" : "🌙";
  }

  // Mobile menu toggle
  if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }

  // Close sidebar on link click (mobile)
  sidebar.addEventListener("click", (e) => {
    if (e.target.tagName === "A" && window.innerWidth <= 900) {
      sidebar.classList.remove("open");
    }
  });

  // 2. Keyboard shortcut for search
  window.addEventListener("keydown", (e) => {
    if ((e.key === "/" || (e.ctrlKey && e.key === "k")) && document.activeElement !== searchInput) {
      e.preventDefault();
      searchInput.focus();
    }
  });

  // 3. Routing
  function handleRoute() {
    const hash = window.location.hash.slice(1) || "overview";
    updateActiveNavLink(hash);

    if (hash === "overview") {
      renderOverview();
    } else if (hash === "hardware") {
      renderHardware();
    } else if (hash === "examples") {
      renderExamples();
    } else if (hash.startsWith("class-")) {
      const className = hash.replace("class-", "");
      renderClassView(className);
    } else if (hash.startsWith("method-")) {
      const parts = hash.replace("method-", "").split("-");
      const className = parts[0];
      const methodId = hash.replace("method-", "");
      renderClassView(className, methodId);
    } else {
      renderOverview();
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  window.addEventListener("hashchange", handleRoute);

  function updateActiveNavLink(hash) {
    document.querySelectorAll(".nav-link").forEach(link => {
      if (link.getAttribute("href") === `#${hash}` || link.getAttribute("href") === `#class-${hash}`) {
        link.classList.add("active");
      } else {
        link.classList.remove("active");
      }
    });
  }

  // 4. Views
  function renderOverview() {
    contentArea.innerHTML = `
      <div class="breadcrumbs">
        <span>tinyISB Documentation</span> &gt; <span>Overview</span>
      </div>
      <div class="doc-header">
        <div class="doc-title-row">
          <h1 class="doc-title">tinyISB API Reference</h1>
          <span class="doc-tag">C++17</span>
          <span class="doc-tag">TFLM</span>
        </div>
        <div class="header-include">#include &lt;tinyISB.h&gt;</div>
        <p class="doc-lead">
          High-performance C++ inference library for <strong>TensorFlow Lite Micro</strong> on microcontrollers (ARM Cortex-M, Xtensa LX7, RISC-V). 
          Provides a high-level <strong>Scikit-Learn inspired API</strong> with <strong>guaranteed zero dynamic memory allocation</strong> during inference.
        </p>
      </div>

      <div class="metrics-row">
        <div class="metric-card">
          <span class="metric-label">Heap Allocation</span>
          <span class="metric-value highlight-green">0 Bytes</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Memory Safety</span>
          <span class="metric-value">Deterministic</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Target Architectures</span>
          <span class="metric-value">ARM / Xtensa / RISC-V</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">API Style</span>
          <span class="metric-value">Scikit-Learn</span>
        </div>
      </div>

      <h2 class="section-subtitle">Core Estimator Interfaces</h2>
      <p style="margin-bottom: 16px; color: var(--text-secondary);">
        tinyISB exposes two primary estimators designed after Scikit-Learn conventions:
      </p>

      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 32px;">
        <div class="metric-card" style="cursor: pointer;" onclick="location.hash='#class-Classifier'">
          <span class="doc-tag" style="width: fit-content; margin-bottom: 8px;">CLASSIFIER</span>
          <h3 style="font-size: 1.2rem; margin-bottom: 8px; color: var(--accent);">tinyisb::Classifier</h3>
          <p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 12px;">
            Multi-class inference with <code>predict()</code> for argmax class selection and <code>predict_proba()</code> for normalized Softmax probabilities.
          </p>
          <span style="font-size: 0.85rem; font-weight: 600; color: var(--accent);">View Methods &rarr;</span>
        </div>

        <div class="metric-card" style="cursor: pointer;" onclick="location.hash='#class-Regressor'">
          <span class="doc-tag" style="width: fit-content; margin-bottom: 8px;">REGRESSOR</span>
          <h3 style="font-size: 1.2rem; margin-bottom: 8px; color: var(--accent);">tinyisb::Regressor</h3>
          <p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 12px;">
            Continuous scalar and vector regression with automatic INT8 output de-quantization back to real-world engineering units.
          </p>
          <span style="font-size: 0.85rem; font-weight: 600; color: var(--accent);">View Methods &rarr;</span>
        </div>
      </div>

      <h2 class="section-subtitle">Quick Start Example</h2>
      <div class="syntax-container">
        <pre class="syntax-box"><code>#include &lt;Arduino.h&gt;

#if defined(ARDUINO_ARDUINO_NANO33BLE)
  #include &lt;TensorFlowLite.h&gt;
#else
  #include &lt;tensorflow-runtime-universal.h&gt;
#endif

#include &lt;tinyISB.h&gt;
#include "model_data.h"

// 1. Declare external contiguous Tensor Arena in static RAM
constexpr size_t kArenaSize = 4 * 1024; // 4 KB static buffer
alignas(16) static uint8_t tensor_arena[kArenaSize];

static tinyisb::Classifier classifier;

void setup() {
    Serial.begin(115200);

    // 2. Initialize classifier with static memory arena
    tinyisb::Status status = classifier.begin(model_data, tensor_arena, kArenaSize);
    if (status != tinyisb::Status::OK) {
        Serial.println("Init failed!");
        while(1);
    }
}

void loop() {
    float features[4] = {0.9375f, -0.3574f, 0.3241f, 0.4415f};

    // 3. Scikit-Learn style argmax prediction
    int predicted_class = classifier.predict(features, 4);

    // 4. Normalized class probability [0.0, 1.0]
    float confidence = classifier.predict_proba(features, predicted_class);

    Serial.print("Class: "); Serial.print(predicted_class);
    Serial.print(" | Conf: "); Serial.print(confidence * 100.0f); Serial.println("%");
    delay(1000);
}</code></pre>
        <button class="btn-copy" onclick="copyCode(this)">Copy</button>
      </div>
    `;
  }

  function renderClassView(className, targetMethodId = null) {
    const classData = TINYISB_API_DATA.classes[className];
    if (!classData) {
      renderOverview();
      return;
    }

    let methodsHtml = "";
    if (classData.methods && classData.methods.length > 0) {
      methodsHtml = classData.methods.map(method => `
        <div class="method-card" id="${method.id}">
          <div class="method-header">
            <span class="method-name">${method.name}</span>
            <span class="nav-badge">METHOD</span>
          </div>
          <div class="method-body">
            <p style="margin-bottom: 16px; font-weight: 500;">${method.brief}</p>
            
            <div class="syntax-container">
              <pre class="syntax-box"><code>${escapeHtml(method.signature)}</code></pre>
              <button class="btn-copy" onclick="copyCode(this)">Copy</button>
            </div>

            <p style="color: var(--text-secondary); margin-bottom: 20px;">${method.details}</p>

            ${method.params && method.params.length > 0 ? `
              <h4 class="section-subtitle">Parameters</h4>
              <table class="params-table">
                <thead>
                  <tr>
                    <th>Dir</th>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Description &amp; Constraints</th>
                  </tr>
                </thead>
                <tbody>
                  ${method.params.map(p => `
                    <tr>
                      <td><span class="dir-badge ${p.direction === '[in]' ? 'dir-in' : (p.direction === '[out]' ? 'dir-out' : 'dir-inout')}">${p.direction}</span></td>
                      <td class="param-name">${p.name}</td>
                      <td class="param-type">${escapeHtml(p.type)}</td>
                      <td>${p.description} ${p.constraints ? `<em>(${p.constraints})</em>` : ''}</td>
                    </tr>
                  `).join("")}
                </tbody>
              </table>
            ` : ''}

            <h4 class="section-subtitle">Return Value</h4>
            <div style="background: var(--bg-secondary); border: 1px solid var(--border-color); padding: 12px 16px; border-radius: var(--radius-md); margin-bottom: 20px;">
              <code style="font-family: var(--font-mono); color: var(--accent); font-weight: 700;">${escapeHtml(method.returns.type)}</code>
              <p style="margin-top: 6px; font-size: 0.9rem; color: var(--text-secondary);">${method.returns.description}</p>
              ${method.returns.errorValues ? `
                <ul style="margin: 8px 0 0 20px; font-size: 0.85rem; color: var(--danger);">
                  ${method.returns.errorValues.map(err => `<li>${err}</li>`).join("")}
                </ul>
              ` : ''}
            </div>

            <h4 class="section-subtitle">Complexity &amp; Memory Safety</h4>
            <div class="metrics-row" style="margin: 12px 0 20px 0;">
              <div class="metric-card" style="padding: 12px;">
                <span class="metric-label">Heap Allocation</span>
                <span class="metric-value highlight-green" style="font-size: 1rem;">${method.complexity.heapAlloc}</span>
              </div>
              <div class="metric-card" style="padding: 12px;">
                <span class="metric-label">Time Complexity</span>
                <span class="metric-value" style="font-size: 1rem;">${method.complexity.timeComplexity}</span>
              </div>
              <div class="metric-card" style="padding: 12px;">
                <span class="metric-label">Stack Footprint</span>
                <span class="metric-value" style="font-size: 1rem;">${method.complexity.stackFootprint}</span>
              </div>
            </div>

            ${method.preconditions && method.preconditions.length > 0 ? `
              <div class="alert-box alert-warning">
                <div class="alert-title">Preconditions</div>
                <ul style="margin-left: 20px;">
                  ${method.preconditions.map(c => `<li>${c}</li>`).join("")}
                </ul>
              </div>
            ` : ''}

            ${method.example ? `
              <h4 class="section-subtitle">Usage Example</h4>
              <div class="syntax-container">
                <pre class="syntax-box"><code>${escapeHtml(method.example)}</code></pre>
                <button class="btn-copy" onclick="copyCode(this)">Copy</button>
              </div>
            ` : ''}
          </div>
        </div>
      `).join("");
    }

    let valuesHtml = "";
    if (classData.values) {
      valuesHtml = `
        <h3 class="section-subtitle">Enumeration Values</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th>Status Enumerator</th>
              <th>Numeric Value</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            ${classData.values.map(v => `
              <tr>
                <td><code style="font-family: var(--font-mono); color: var(--accent); font-weight: bold;">${v.name}</code></td>
                <td><code>${v.value}</code></td>
                <td>${v.description}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }

    let propertiesHtml = "";
    if (classData.properties) {
      propertiesHtml = `
        <h3 class="section-subtitle">Public Member Fields</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th>Field Name</th>
              <th>Type</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            ${classData.properties.map(p => `
              <tr>
                <td style="font-family: var(--font-mono); font-weight: bold;">${p.name}</td>
                <td><code>${escapeHtml(p.type)}</code></td>
                <td>${p.description}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }

    contentArea.innerHTML = `
      <div class="breadcrumbs">
        <span>tinyISB Documentation</span> &gt; <span>${classData.category}</span> &gt; <span>${classData.name}</span>
      </div>
      <div class="doc-header">
        <div class="doc-title-row">
          <h1 class="doc-title">${classData.name}</h1>
          <span class="doc-tag">${classData.category}</span>
        </div>
        <div class="header-include">${classData.header}</div>
        <p class="doc-lead">${classData.description}</p>
      </div>

      ${propertiesHtml}
      ${valuesHtml}

      ${classData.methods ? `
        <h2 style="margin: 32px 0 20px 0; font-size: 1.4rem; font-weight: 700;">Methods</h2>
        ${methodsHtml}
      ` : ''}
    `;

    if (targetMethodId) {
      setTimeout(() => {
        const el = document.getElementById(targetMethodId);
        if (el) el.scrollIntoView({ behavior: "smooth" });
      }, 100);
    }
  }

  function renderHardware() {
    contentArea.innerHTML = `
      <div class="breadcrumbs">
        <span>tinyISB Documentation</span> &gt; <span>Target Platforms</span>
      </div>
      <div class="doc-header">
        <h1 class="doc-title">Target Microcontrollers &amp; Compiler Macros</h1>
        <p class="doc-lead">
          tinyISB provides guaranteed cross-platform compilation across ARM Cortex-M, Xtensa, and RISC-V architectures through compile-time macro resolution.
        </p>
      </div>

      <div class="alert-box">
        <div class="alert-title">Zero-Touch Compilation</div>
        When targeting a microcontroller, the board core defines preprocessor macros automatically. You never need to modify the <code>#if defined(...)</code> headers manually in your sketch.
      </div>

      <table class="data-table">
        <thead>
          <tr>
            <th>Board</th>
            <th>Architecture &amp; Core</th>
            <th>Automatic Compiler Macros</th>
            <th>Arduino CLI FQBN</th>
            <th>Internal OpResolver</th>
          </tr>
        </thead>
        <tbody>
          ${TINYISB_API_DATA.hardwareTargets.map(h => `
            <tr>
              <td><strong>${h.board}</strong></td>
              <td>${h.arch}<br><small style="color: var(--text-muted);">${h.core}</small></td>
              <td><code>${h.macros}</code></td>
              <td><code>${h.fqbn}</code></td>
              <td><code>${h.resolver}</code></td>
            </tr>
          `).join("")}
        </tbody>
      </table>

      <h3 class="section-subtitle">Build Commands via arduino-cli</h3>
      <div class="syntax-container">
        <pre class="syntax-box"><code># Raspberry Pi Pico (RP2040)
arduino-cli compile -b rp2040:rp2040:rpipico examples/BasicClassification

# Raspberry Pi Pico 2 (RP2350 Cortex-M33)
arduino-cli compile -b rp2040:rp2040:rpipico2 examples/BasicClassification

# ESP32-S3 Dev Module (Xtensa LX7)
arduino-cli compile -b esp32:esp32:esp32s3 examples/BasicClassification

# ESP32-C6 Dev Module (RISC-V)
arduino-cli compile -b esp32:esp32:esp32c6 examples/BasicClassification

# Arduino Nano 33 BLE Sense (nRF52840 Cortex-M4F)
arduino-cli compile -b arduino:mbed_nano:nano33ble examples/BasicClassification</code></pre>
        <button class="btn-copy" onclick="copyCode(this)">Copy</button>
      </div>
    `;
  }

  function renderExamples() {
    contentArea.innerHTML = `
      <div class="breadcrumbs">
        <span>tinyISB Documentation</span> &gt; <span>Examples</span>
      </div>
      <div class="doc-header">
        <h1 class="doc-title">Library Examples &amp; Tutorials</h1>
        <p class="doc-lead">
          Four production-grade examples covering different ML paradigms (Feed-Forward MLP, 1D Convolutional Neural Networks, Knowledge Distillation, and Continuous Parameter Regression).
        </p>
      </div>

      <div style="display: grid; grid-template-columns: 1fr; gap: 24px;">
        ${TINYISB_API_DATA.examples.map(ex => `
          <div class="method-card" style="margin-bottom: 0;">
            <div class="method-header">
              <span class="method-name" style="font-size: 1.1rem; color: var(--text-primary);">${ex.title}</span>
              <span class="nav-badge">${ex.id}</span>
            </div>
            <div class="method-body">
              <p style="color: var(--text-secondary); margin-bottom: 16px;">${ex.description}</p>
              
              <div class="metrics-row" style="margin: 0 0 16px 0;">
                <div class="metric-card" style="padding: 10px;">
                  <span class="metric-label">Topology</span>
                  <span style="font-size: 0.9rem; font-family: var(--font-mono); font-weight: bold;">${ex.model}</span>
                </div>
                <div class="metric-card" style="padding: 10px;">
                  <span class="metric-label">Flash Size</span>
                  <span style="font-size: 0.9rem; font-family: var(--font-mono); font-weight: bold;">${ex.flashSize}</span>
                </div>
                <div class="metric-card" style="padding: 10px;">
                  <span class="metric-label">Tensor Arena RAM</span>
                  <span style="font-size: 0.9rem; font-family: var(--font-mono); font-weight: bold;">${ex.arenaRAM}</span>
                </div>
                <div class="metric-card" style="padding: 10px;">
                  <span class="metric-label">Latency (RP2040)</span>
                  <span class="metric-value highlight-green" style="font-size: 0.95rem;">${ex.latencyRP2040}</span>
                </div>
              </div>

              <div style="display: flex; gap: 12px; align-items: center;">
                <code style="font-family: var(--font-mono); font-size: 0.85rem; color: var(--accent);">${ex.link}</code>
              </div>
            </div>
          </div>
        `).join("")}
      </div>
    `;
  }

  // 5. Search Functionality
  searchInput.addEventListener("input", (e) => {
    const query = e.target.value.toLowerCase().trim();
    if (!query) {
      handleRoute();
      return;
    }

    const matches = [];

    // Search classes & methods
    Object.keys(TINYISB_API_DATA.classes).forEach(className => {
      const cls = TINYISB_API_DATA.classes[className];
      if (cls.name.toLowerCase().includes(query) || cls.brief.toLowerCase().includes(query)) {
        matches.push({
          type: "Class",
          title: cls.name,
          desc: cls.brief,
          hash: `#class-${className}`
        });
      }

      if (cls.methods) {
        cls.methods.forEach(m => {
          if (m.name.toLowerCase().includes(query) || m.brief.toLowerCase().includes(query) || m.signature.toLowerCase().includes(query)) {
            matches.push({
              type: "Method",
              title: `${cls.name}::${m.name}()`,
              desc: m.brief,
              hash: `#method-${className}-${m.id}`
            });
          }
        });
      }
    });

    // Render search results
    contentArea.innerHTML = `
      <div class="doc-header">
        <h1 class="doc-title">Search Results</h1>
        <p class="doc-lead">Found ${matches.length} matches for "${escapeHtml(query)}"</p>
      </div>

      <div style="display: flex; flex-direction: column; gap: 16px;">
        ${matches.length > 0 ? matches.map(m => `
          <div class="metric-card" style="cursor: pointer;" onclick="location.hash='${m.hash}'">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <span class="nav-badge">${m.type}</span>
              <span style="font-size: 0.8rem; color: var(--accent);">View &rarr;</span>
            </div>
            <h3 style="font-size: 1.1rem; color: var(--accent); font-family: var(--font-mono);">${m.title}</h3>
            <p style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 4px;">${m.desc}</p>
          </div>
        `).join("") : `
          <p style="color: var(--text-muted); font-size: 1.1rem;">No matching symbols or methods found. Try searching for "predict", "begin", "arena", "status", etc.</p>
        `}
      </div>
    `;
  });

  // Helper functions
  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  window.copyCode = function(button) {
    const pre = button.previousElementSibling;
    const text = pre.textContent;
    navigator.clipboard.writeText(text).then(() => {
      button.textContent = "Copied!";
      setTimeout(() => { button.textContent = "Copy"; }, 2000);
    });
  };

  // Initial Route Load
  handleRoute();
});
