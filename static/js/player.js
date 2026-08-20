/* LLaDA-UI project page — frame-sequence player.
 *
 * Powers the denoising showcase and the agent-trace animations from the
 * report's recorded frames. Markup contract:
 *   <div class="player" data-player
 *        data-pattern="static/images/denoise/frame-{i}.png"
 *        data-frames="75" data-fps="3" data-autoplay="1"
 *        data-label="Denoising sequence"></div>
 * Frames are 1-indexed. Playback loops, starts when the player scrolls into
 * view (unless the visitor prefers reduced motion or pressed pause), and
 * pauses off screen. All frames preload once the player first becomes
 * visible so scrubbing is instant.
 */

(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var ICONS = {
    play: '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4 2.5v11l9-5.5-9-5.5z"/></svg>',
    pause: '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z"/></svg>',
    prev: '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M11.5 2.5v11L4 8l7.5-5.5zM3 2.5h1.6v11H3z"/></svg>',
    next: '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4.5 2.5v11L12 8 4.5 2.5zM11.4 2.5H13v11h-1.6z"/></svg>'
  };

  function initPlayer(root) {
    var pattern = root.dataset.pattern;
    var total = parseInt(root.dataset.frames, 10);
    var fps = parseFloat(root.dataset.fps) || 2;
    var wantAuto = root.dataset.autoplay === "1" && !reduceMotion;
    if (!pattern || !total) return;

    var src = function (i) { return pattern.replace("{i}", String(i)); };

    var frameBox = document.createElement("div");
    frameBox.className = "frame-box";
    var img = document.createElement("img");
    img.src = src(1);
    img.alt = root.dataset.label || "Animation frame";
    frameBox.appendChild(img);

    var controls = document.createElement("div");
    controls.className = "controls";

    function button(icon, label) {
      var b = document.createElement("button");
      b.type = "button";
      b.innerHTML = ICONS[icon];
      b.setAttribute("aria-label", label);
      return b;
    }

    var playBtn = button(wantAuto ? "pause" : "play", "Play or pause");
    var prevBtn = button("prev", "Previous frame");
    var nextBtn = button("next", "Next frame");

    var scrub = document.createElement("input");
    scrub.type = "range";
    scrub.className = "scrub";
    scrub.min = "1";
    scrub.max = String(total);
    scrub.value = "1";
    scrub.setAttribute("aria-label", (root.dataset.label || "Animation") + " frame");

    var counter = document.createElement("span");
    counter.className = "counter";

    controls.appendChild(playBtn);
    controls.appendChild(prevBtn);
    controls.appendChild(nextBtn);
    controls.appendChild(scrub);
    controls.appendChild(counter);

    root.appendChild(frameBox);
    root.appendChild(controls);

    var frame = 1;
    var timer = null;
    var userPaused = !wantAuto;
    var preloaded = false;

    function render() {
      img.src = src(frame);
      scrub.value = String(frame);
      scrub.style.setProperty("--fill", ((frame - 1) / (total - 1) * 100) + "%");
      counter.textContent = frame + " / " + total;
    }

    function setPlaying(on) {
      if (on && !timer) {
        timer = setInterval(function () {
          frame = frame % total + 1;
          render();
        }, 1000 / fps);
        playBtn.innerHTML = ICONS.pause;
      } else if (!on && timer) {
        clearInterval(timer);
        timer = null;
        playBtn.innerHTML = ICONS.play;
      }
    }

    function preload() {
      if (preloaded) return;
      preloaded = true;
      for (var i = 2; i <= total; i++) {
        var pre = new Image();
        pre.src = src(i);
      }
    }

    playBtn.addEventListener("click", function () {
      userPaused = !!timer;
      preload();
      setPlaying(!timer);
    });
    prevBtn.addEventListener("click", function () {
      userPaused = true;
      setPlaying(false);
      frame = (frame - 2 + total) % total + 1;
      render();
    });
    nextBtn.addEventListener("click", function () {
      userPaused = true;
      setPlaying(false);
      frame = frame % total + 1;
      render();
    });
    scrub.addEventListener("input", function () {
      userPaused = true;
      setPlaying(false);
      frame = parseInt(scrub.value, 10) || 1;
      render();
    });

    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            preload();
            if (!userPaused) setPlaying(true);
          } else {
            setPlaying(false);
          }
        });
      }, { threshold: 0.25 });
      io.observe(root);
    } else {
      preload();
      if (!userPaused) setPlaying(true);
    }

    render();
  }

  document.querySelectorAll("[data-player]").forEach(initPlayer);
})();
