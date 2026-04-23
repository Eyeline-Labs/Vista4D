window.HELP_IMPROVE_VIDEOJS = false;

// Minimum time (ms) before a carousel auto-advances. The actual delay is
// ceil(minTime / videoDuration) × videoDuration, so each video always
// finishes at least one full loop and the total wait is >= this value.
var CAROUSEL_AUTOPLAY_MIN = 5000;

// Load a video's src from its data-src attribute (lazy loading).
function loadVideo(videoEl) {
    if (videoEl.dataset.loaded) return;
    videoEl.dataset.loaded = '1';
    var source = videoEl.querySelector('source[data-src]');
    if (source) {
        source.src = source.dataset.src;
        videoEl.load();
    }
}

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Carousel initialization
function initCarousels() {
    document.querySelectorAll('.carousel').forEach(function(carousel) {
        var track = carousel.querySelector('.carousel-track');
        var items = Array.from(track.children);
        var prevBtn = carousel.querySelector('.carousel-prev');
        var nextBtn = carousel.querySelector('.carousel-next');
        var dotsContainer = carousel.querySelector('.carousel-dots');
        var currentIndex = 0;
        var autoplayTimer = null;
        var pendingMetaCb = null;
        var isTransitioning = false;
        var isTwoUp = carousel.classList.contains('carousel-two-up');

        function getSlidesPerView() {
            if (!isTwoUp) return 1;
            return window.innerWidth > 768 ? 2 : 1;
        }

        var maxSlidesPerView = isTwoUp ? 2 : 1;

        // Clone items for seamless infinite scrolling:
        // - Prepend clone of last item (for wrapping left)
        // - Append clones of first N items (for wrapping right + filling two-up view)
        var cloneBefore = items[items.length - 1].cloneNode(true);
        cloneBefore.setAttribute('aria-hidden', 'true');
        track.insertBefore(cloneBefore, track.firstChild);

        for (var c = 0; c < maxSlidesPerView; c++) {
            var cloneAfter = items[c].cloneNode(true);
            cloneAfter.setAttribute('aria-hidden', 'true');
            track.appendChild(cloneAfter);
        }

        // All track children including clones (for video management)
        var allTrackChildren = Array.from(track.children);

        // Stop all carousel videos from auto-playing on page load.
        // src is not set yet (lazy-loaded via data-src), so use preload="none".
        allTrackChildren.forEach(function(item) {
            var v = item.querySelector('video');
            if (v) {
                v.removeAttribute('autoplay');
                v.setAttribute('preload', 'none');
                v.pause();
            }
        });

        // Play only the currently visible videos from the start, pause the rest.
        // Skip if the carousel is hidden (e.g., inside a collapsed section).
        function activateVideos() {
            if (!carousel.offsetParent) return;

            var slidesPerView = getSlidesPerView();
            var startPos = currentIndex + 1; // +1 for prepended clone

            allTrackChildren.forEach(function(item, domIdx) {
                var v = item.querySelector('video');
                if (!v) return;
                if (domIdx >= startPos && domIdx < startPos + slidesPerView) {
                    loadVideo(v);
                    v.setAttribute('preload', 'auto');
                    v.currentTime = 0;
                    v.play();
                } else {
                    v.pause();
                    v.setAttribute('preload', 'none');
                }
            });

            // Preload the next and previous real slides so navigation feels instant.
            [currentIndex + 1, currentIndex - 1].forEach(function(i) {
                var realIdx = ((i % items.length) + items.length) % items.length;
                var item = allTrackChildren[realIdx + 1]; // +1 for prepended clone
                if (item) {
                    var v = item.querySelector('video');
                    if (v) loadVideo(v);
                }
            });
        }

        // Offset by 1 to skip the prepended clone
        function setTransform(index) {
            var stepPercent = 100 / getSlidesPerView();
            var pos = (index + 1) * stepPercent;
            track.style.transform = 'translateX(-' + pos + '%)';
        }

        // Set initial position without animation (activation deferred to IntersectionObserver)
        track.style.transition = 'none';
        setTransform(0);
        track.offsetHeight; // force reflow
        track.style.transition = '';

        // Create dots (one per real item)
        items.forEach(function(_, i) {
            var dot = document.createElement('button');
            dot.className = 'carousel-dot' + (i === 0 ? ' active' : '');
            dot.addEventListener('click', function() { goTo(i); });
            dotsContainer.appendChild(dot);
        });
        var dots = Array.from(dotsContainer.children);

        function updateDots() {
            dots.forEach(function(d, i) {
                d.classList.toggle('active', i === currentIndex);
            });
        }

        function goTo(index) {
            if (isTransitioning) return;

            if (index < 0) {
                // Wrapping left: animate to the prepended clone, then snap
                isTransitioning = true;
                setTransform(-1);
                currentIndex = items.length - 1;
                updateDots();
                resetAutoplay();
                return;
            }

            if (index >= items.length) {
                // Wrapping right: animate to the appended clone, then snap
                isTransitioning = true;
                setTransform(items.length);
                currentIndex = 0;
                updateDots();
                resetAutoplay();
                return;
            }

            currentIndex = index;
            setTransform(currentIndex);
            updateDots();
            activateVideos();
            resetAutoplay();
        }

        // After animating to a clone, instantly snap to the real position
        track.addEventListener('transitionend', function(e) {
            if (e.propertyName === 'transform' && isTransitioning) {
                isTransitioning = false;
                track.style.transition = 'none';
                setTransform(currentIndex);
                track.offsetHeight; // force reflow before restoring transition
                track.style.transition = '';
                activateVideos();
            }
        });

        prevBtn.addEventListener('click', function() { goTo(currentIndex - 1); });
        nextBtn.addEventListener('click', function() { goTo(currentIndex + 1); });

        // Keyboard navigation
        carousel.setAttribute('tabindex', '0');
        carousel.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft') goTo(currentIndex - 1);
            if (e.key === 'ArrowRight') goTo(currentIndex + 1);
        });

        // Pause autoplay on hover
        carousel.addEventListener('mouseenter', function() {
            cleanupAutoplay();
        });
        carousel.addEventListener('mouseleave', function() {
            resetAutoplay();
        });

        // Compute autoplay delay: N full loops of the current video where
        // N = max(1, ceil(CAROUSEL_AUTOPLAY_MIN / duration)).
        // Returns null if video duration is not yet available.
        function getAutoplayDelay() {
            var video = items[currentIndex] && items[currentIndex].querySelector('video');
            if (video && isFinite(video.duration) && video.duration > 0) {
                var durationMs = video.duration * 1000;
                var n = Math.max(1, Math.ceil(CAROUSEL_AUTOPLAY_MIN / durationMs));
                return n * durationMs;
            }
            return null;
        }

        function cleanupAutoplay() {
            if (autoplayTimer) clearTimeout(autoplayTimer);
            autoplayTimer = null;
            if (pendingMetaCb) {
                pendingMetaCb.video.removeEventListener('loadedmetadata', pendingMetaCb.fn);
                pendingMetaCb = null;
            }
        }

        function resetAutoplay() {
            cleanupAutoplay();

            // Don't auto-advance if carousel is hidden (collapsed section)
            if (!carousel.offsetParent) return;

            var delay = getAutoplayDelay();
            if (delay !== null) {
                autoplayTimer = setTimeout(function() {
                    goTo(currentIndex + 1);
                }, delay);
            } else {
                // Duration not available yet (video still loading);
                // wait for metadata then set the proper timer
                var video = items[currentIndex] && items[currentIndex].querySelector('video');
                if (video) {
                    var fn = function() {
                        video.removeEventListener('loadedmetadata', fn);
                        pendingMetaCb = null;
                        resetAutoplay();
                    };
                    pendingMetaCb = { video: video, fn: fn };
                    video.addEventListener('loadedmetadata', fn);
                }
                // Fallback in case metadata never loads
                autoplayTimer = setTimeout(function() {
                    if (pendingMetaCb) {
                        pendingMetaCb.video.removeEventListener('loadedmetadata', pendingMetaCb.fn);
                        pendingMetaCb = null;
                    }
                    goTo(currentIndex + 1);
                }, CAROUSEL_AUTOPLAY_MIN);
            }
        }

        // Update transform on resize when slidesPerView changes
        if (isTwoUp) {
            var lastSlidesPerView = getSlidesPerView();
            window.addEventListener('resize', function() {
                var newSlidesPerView = getSlidesPerView();
                if (newSlidesPerView !== lastSlidesPerView) {
                    lastSlidesPerView = newSlidesPerView;
                    track.style.transition = 'none';
                    setTransform(currentIndex);
                    track.offsetHeight;
                    track.style.transition = '';
                    activateVideos();
                }
            });
        }

        // Custom events for collapsible sections
        carousel.addEventListener('carousel-activate', function() {
            activateVideos();
            resetAutoplay();
        });
        carousel.addEventListener('carousel-deactivate', function() {
            cleanupAutoplay();
            allTrackChildren.forEach(function(item) {
                var v = item.querySelector('video');
                if (v) v.pause();
            });
        });

        // Activate the carousel only when it scrolls into view; pause when it leaves.
        if ('IntersectionObserver' in window) {
            var carouselObserver = new IntersectionObserver(function(entries) {
                entries.forEach(function(entry) {
                    if (entry.isIntersecting) {
                        activateVideos();
                        resetAutoplay();
                    } else {
                        cleanupAutoplay();
                        allTrackChildren.forEach(function(item) {
                            var v = item.querySelector('video');
                            if (v) v.pause();
                        });
                    }
                });
            }, { rootMargin: '100px 0px' });
            carouselObserver.observe(carousel);
        } else {
            // Fallback for browsers without IntersectionObserver
            activateVideos();
            resetAutoplay();
        }
    });
}

// Toggle a collapsible section and activate/deactivate its carousels
function toggleCollapsible(header) {
    var content = header.nextElementSibling;
    var icon = header.querySelector('.collapsible-icon');
    var hint = header.querySelector('.collapsible-hint');
    var isShowing = content.classList.toggle('show');
    icon.classList.toggle('open', isShowing);
    if (hint) hint.textContent = isShowing ? 'Click to collapse' : 'Click to expand';

    content.querySelectorAll('.carousel').forEach(function(c) {
        c.dispatchEvent(new Event(isShowing ? 'carousel-activate' : 'carousel-deactivate'));
    });
}

// Hero video: starts muted (autoplay), fades volume in/out on hover.
//
// Browsers (Chrome in particular) treat `mouseenter` as a non-qualifying user
// gesture for unmuted video playback. Calling `video.muted = false` or
// `video.play()` from mouseenter causes Chrome to pause the video. To avoid
// this, we unlock audio on the first real user gesture (pointerdown/keydown),
// which DO qualify. After that, mouseenter/mouseleave only fade the volume —
// no muted changes or play() calls during hover.
function initHeroVideoHover() {
    var video = document.getElementById('tree');
    if (!video) return;

    var fadeRaf = null;
    var audioUnlocked = false;

    function cancelFade() {
        if (fadeRaf !== null) {
            cancelAnimationFrame(fadeRaf);
            fadeRaf = null;
        }
    }

    function fadeVolume(from, to, duration) {
        cancelFade();
        var startTime = null;
        function step(timestamp) {
            if (!startTime) startTime = timestamp;
            var t = Math.min((timestamp - startTime) / duration, 1);
            var eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
            video.volume = from + (to - from) * eased;
            if (t < 1) {
                fadeRaf = requestAnimationFrame(step);
            } else {
                fadeRaf = null;
            }
        }
        fadeRaf = requestAnimationFrame(step);
    }

    // Called from a qualifying user gesture (pointerdown/keydown). Sets
    // muted=false and volume=0 while the gesture is active so the browser
    // allows unmuted playback. After this, mouseenter/mouseleave just fade
    // the volume — no further muted changes needed.
    function unlockAudio() {
        if (audioUnlocked) return;
        video.volume = 0;
        video.muted = false;
        video.play().then(function() {
            audioUnlocked = true;
            document.removeEventListener('pointerdown', unlockAudio);
            document.removeEventListener('keydown', unlockAudio);
            // If the cursor is already over the video, start the fade-in now.
            if (video.matches(':hover')) fadeVolume(0, 1, 500);
        }).catch(function() {
            // Shouldn't happen inside a real gesture, but restore just in case.
            video.muted = true;
            if (video.paused) video.play().catch(function() {});
        });
    }

    document.addEventListener('pointerdown', unlockAudio);
    document.addEventListener('keydown', unlockAudio);

    video.addEventListener('mouseenter', function() {
        if (!audioUnlocked) return;
        cancelFade();
        fadeVolume(video.volume, 1, 500);
    });

    video.addEventListener('mouseleave', function() {
        if (!audioUnlocked) return;
        cancelFade();
        fadeVolume(video.volume, 0, 500);
    });
}

document.addEventListener('DOMContentLoaded', function() {
    initCarousels();
    initHeroVideoHover();
});
