document.addEventListener('DOMContentLoaded', () => {
    // 1. Create Triangle Mesh Overlay
    const overlay = document.createElement('div');
    overlay.classList.add('transition-overlay');
    document.body.appendChild(overlay);

    // Grid size (10x10)
    const rows = 10;
    const cols = 10;
    const totalTriangles = rows * cols;

    // Create triangles
    for (let i = 0; i < totalTriangles; i++) {
        const triangle = document.createElement('div');
        triangle.classList.add('transition-triangle');
        // Add random slight delay for organic feel
        triangle.style.transitionDelay = `${Math.random() * 0.2}s`;
        overlay.appendChild(triangle);
    }

    const triangles = document.querySelectorAll('.transition-triangle');

    // 2. Animate IN (Page Load) - Turn off scaling to reveal page
    // Small timeout to ensure DOM is ready
    setTimeout(() => {
        triangles.forEach(t => {
            t.classList.remove('active'); // Scale down to 0
        });
    }, 100);

    // Initialize with active class so they cover screen, then remove it
    triangles.forEach(t => t.classList.add('active'));


    // 3. Handle Navigation Clicks
    document.querySelectorAll('a').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            const target = this.getAttribute('target');

            // Ignore external links, anchors to same page, or special links
            if (target === '_blank' || href.startsWith('#') || href.startsWith('mailto:') || href.endsWith('.pdf')) {
                return; // Default behavior
            }

            e.preventDefault();

            // Animate OUT (Page Leave) - Scale up to cover screen
            triangles.forEach(t => {
                // Randomize delay again for closing animation
                t.style.transitionDelay = `${Math.random() * 0.2}s`;
                t.classList.add('active');
            });

            // Wait for animation to finish, then navigate
            setTimeout(() => {
                window.location.href = href;
            }, 600); // 0.4s transition + 0.2s max delay
        });
    });

    // 4. Scroll Animation (IntersectionObserver)
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    document.querySelectorAll('.fade-in-up').forEach(el => observer.observe(el));
});
