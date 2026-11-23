/* Progressive enhancement: links and content work without JavaScript. */
(() => {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  function init() {
    document.querySelectorAll("[data-carousel]").forEach(carousel => {
      const slides = [...carousel.querySelectorAll(".carousel-slide")];
      let index = 0, timer, paused = reducedMotion.matches, hovered = false, focused = false;
      const pauseButton = carousel.querySelector("[data-pause]");
      const counter = carousel.querySelector("[data-counter]");
      function show(next) {
        index = (next + slides.length) % slides.length;
        slides.forEach((slide, n) => {
          slide.classList.toggle("is-active", n === index);
          slide.setAttribute("aria-hidden", String(n !== index));
        });
        counter.textContent = String(index + 1).padStart(2, "0") + " / " + slides.length;
      }
      function schedule() {
        clearInterval(timer);
        if (!paused && !hovered && !focused && !document.hidden)
          timer = setInterval(() => show(index + 1), 5000);
      }
      function updatePause() {
        pauseButton.textContent = paused ? "播放" : "暂停";
        pauseButton.setAttribute("aria-label", paused ? "开启自动轮播" : "暂停自动轮播");
        schedule();
      }
      carousel.querySelector("[data-prev]").onclick = () => { show(index - 1); schedule(); };
      carousel.querySelector("[data-next]").onclick = () => { show(index + 1); schedule(); };
      pauseButton.onclick = () => { paused = !paused; updatePause(); };
      carousel.onmouseenter = () => { hovered = true; schedule(); };
      carousel.onmouseleave = () => { hovered = false; schedule(); };
      carousel.onfocusin = () => { focused = true; schedule(); };
      carousel.onfocusout = event => { focused = carousel.contains(event.relatedTarget); schedule(); };
      carousel.onkeydown = event => {
        if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
          event.preventDefault();
          show(index + (event.key === "ArrowRight" ? 1 : -1));
          schedule();
        }
      };
      let touchX;
      carousel.addEventListener("touchstart", event => { touchX = event.changedTouches[0].clientX; }, { passive: true });
      carousel.addEventListener("touchend", event => {
        const distance = event.changedTouches[0].clientX - touchX;
        if (Math.abs(distance) > 45) { show(index + (distance < 0 ? 1 : -1)); schedule(); }
      }, { passive: true });
      document.addEventListener("visibilitychange", schedule);
      reducedMotion.addEventListener("change", () => { paused = reducedMotion.matches; updatePause(); });
      show(0);
      updatePause();
      carousel.classList.add("is-enhanced");
    });
    const gallery = document.querySelector(".gallery-page");
    if (gallery) {
      const items = [...gallery.querySelectorAll(".photo-item")];
      const filters = [...gallery.querySelectorAll("[data-filter]")];
      filters.forEach(button => {
        button.onclick = () => {
          filters.forEach(filter => {
            filter.classList.toggle("is-active", filter === button);
            filter.setAttribute("aria-pressed", String(filter === button));
          });
          let count = 0;
          items.forEach(item => {
            item.hidden = button.dataset.filter !== "all" && item.dataset.category !== button.dataset.filter;
            if (!item.hidden) count++;
          });
          gallery.querySelector(".gallery-count").textContent = count + " 张影像";
        };
      });
      gallery.classList.add("is-enhanced");
    }
    const article = document.querySelector(".md-content__inner");
    if (article && !article.querySelector(".home-page, .gallery-page")) {
      const title = article.querySelector("h1");
      const length = article.textContent.replace(/\s/g, "").length;
      if (title && length > 800) {
        const meta = document.createElement("p");
        meta.className = "reading-meta";
        meta.textContent = "学习笔记 · 约 " + Math.max(1, Math.ceil(length / 450)) + " 分钟阅读";
        title.after(meta);
      }
    }
    const progress = document.createElement("div");
    progress.className = "reading-progress";
    progress.setAttribute("aria-hidden", "true");
    document.body.append(progress);
    const updateProgress = () => {
      const available = document.documentElement.scrollHeight - window.innerHeight;
      progress.style.transform = "scaleX(" + (available > 0 ? Math.min(1, Math.max(0, window.scrollY / available)) : 0) + ")";
    };
    window.addEventListener("scroll", updateProgress, { passive: true });
    window.addEventListener("resize", updateProgress);
    updateProgress();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
