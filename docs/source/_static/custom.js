// Add zoom to images.
document.addEventListener("DOMContentLoaded", () => {
    mediumZoom(".main .content img");
});

// Make external links open new window/tab with security improvements.
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("a.reference.external").forEach(link => {
        link.setAttribute("target", "_blank");
        link.setAttribute("rel", "noopener noreferrer");
    });
});

// Add case-insensitive :icontains selector.
jQuery.expr[':'].icontains = (a, i, m) => jQuery(a).text().toUpperCase().indexOf(m[3].toUpperCase()) >= 0;

// Return all text nodes from an element.
jQuery.fn.textNodes = function () {
    return this.contents().filter(function () {
        return (this.nodeType === Node.TEXT_NODE && this.nodeValue.trim() !== "");
    });
};

// Define allowed render pipelines.
const RENDER_PIPELINES = [
    "birp",
    "hdrp",
    "urp",
];

// Check if the site is localhost or other specific versions.
const isLocalHost = location.hostname === "localhost" || location.hostname === "127.0.0.1" || location.hostname === "";
const isLatest = window.location.pathname.startsWith("/en/latest/") || isLocalHost;
const isStable = window.location.pathname.startsWith("/en/stable/");
const version = isLocalHost ? "latest" : window.location.pathname.split("/").filter(x => x)[1];
const isVersion = !isLatest && !isStable && /^[\dX]+(?:\.[\dX]+)*$/i.test(version);

// Safely update links with render pipeline parameters.
function updateLinksWithRenderPipeline(renderPipeline) {
    document.querySelectorAll("a.reference.internal").forEach(link => {
        const url = new URL(link.href, window.location);
        url.searchParams.set("rp", renderPipeline);
        link.href = url.href;
    });
}

// Safely update the current location without reloading the page.
function updateLocationWithRenderPipeline(renderPipeline) {
    const url = new URL(window.location);
    url.searchParams.set("rp", renderPipeline);
    window.history.pushState({}, "", url);
}

// Add support for RP URL parameter.
document.addEventListener("DOMContentLoaded", () => {
    const renderPipeline = new URLSearchParams(window.location.search).get('rp');
    
    // Validate render pipeline and apply changes.
    if (renderPipeline && RENDER_PIPELINES.includes(renderPipeline.toLowerCase())) {
        $(`.tab-label:icontains("${renderPipeline}")`).click();
        updateLinksWithRenderPipeline(renderPipeline);
    }

    document.querySelectorAll(".tab-label").forEach(tab => {
        tab.addEventListener("click", event => {
            const tabName = event.target.textContent.toLowerCase();
            if (RENDER_PIPELINES.includes(tabName)) {
                updateLinksWithRenderPipeline(tabName);
                updateLocationWithRenderPipeline(tabName);
            }
        });
    });

    // Add "(unreleased)" tag to the latest version if not stable.
    if (isLatest && window.location.pathname.endsWith("history.html")) {
        const headingNode = $("#version h2").textNodes().first();
        headingNode.replaceWith(headingNode.text() + " (unreleased)");
    }

    // Add compatibility notice if on the latest unstable version.
    if (isLatest) {
        const stableUrl = location.href.replace('/latest/', '/stable/');
        const div = document.createElement('div');
        div.classList.add('admonition', 'attention');
        div.innerHTML = `
            <p class="admonition-title">Attention</p>
            <p>
                You are reading the <code class="docutils literal notranslate"><span class="pre">latest</span></code>
                (unstable) version of this documentation, which may document features not available
                or compatible with the latest <em>Crest</em> packages released on the <em>Unity Asset Store</em>.
            </p>
            <p class="last">
                View the <a class="reference" href="${stableUrl}">stable version of this page</a>.
            </p>
        `;
        document.querySelector("article[role='main']").prepend(div);
    }
});

// Redirect to the latest documentation if an unpublished version is detected.
if (typeof isPage404 !== 'undefined' && isPage404 && isVersion) {
    fetch(`/en/${version}/`, { method: "HEAD" })
        .then(response => {
            if (!response.ok) {
                const newUrl = new URL(window.location);
                if (!newUrl.pathname.endsWith("/") && !newUrl.pathname.endsWith(".html")) newUrl.pathname += "/";
                newUrl.href = newUrl.href.replace(`/${version}/`, "/latest/");
                const div = document.createElement('div');
                div.id = "404-admonition";
                div.classList.add('admonition', 'attention');
                div.innerHTML = `
                    <p class="admonition-title">Attention</p>
                    <p class="last">
                        Looks like you are on a version without a published tag.
                        We will redirect you to the latest documentation automatically.
                        If it does not redirect in a few seconds, please click the following link:
                        <a href="${newUrl.href}">${newUrl.href}</a>
                    </p>
                `;
                document.getElementById("404-page-script").insertAdjacentElement("beforebegin", div);
                window.location.replace(newUrl);
            }
        });
}

// Light/Dark mode support for UAS store widgets.
if (window.matchMedia) {
    function applyLightOrDarkMode(isDarkMode) {
        const iframes = document.querySelectorAll("iframe.asset-store");
        if (!iframes.length) return;
        
        const dark = "/widget-wide?";
        const light = "/widget-wide-light?";
        const oldWidget = isDarkMode ? light : dark;
        const newWidget = isDarkMode ? dark : light;

        iframes.forEach(iframe => {
            if (iframe.src.includes(oldWidget)) {
                iframe.src = iframe.src.replace(oldWidget, newWidget);
            }
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        applyLightOrDarkMode(window.matchMedia('(prefers-color-scheme: dark)').matches);
    });

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
        applyLightOrDarkMode(event.matches);
    });
}
