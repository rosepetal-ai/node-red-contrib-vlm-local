/**
 * The debug image a node shows under itself on the editor canvas.
 *
 * Shared by every node in this package that publishes one. It was copied by hand
 * into `inferencer`, `ocr-inferencer` and `vlm-online-inferencer` before this
 * file existed, and the copies had already drifted: the OCR one subscribes to a
 * topic nothing publishes, and it reuses the `inferencer`'s element ids, so had
 * it ever fired the two would have fought over the same DOM nodes.
 *
 * Served by Node-RED from the package's `resources/` directory, at
 * `resources/@rosepetal/node-red-contrib-vlm-local/debug-image-canvas.js`.
 *
 * A node registers itself, in its editor HTML, with:
 *
 *     RosepetalDebugImageCanvas.attach({
 *         topic: 'my-node-debug-image',   // what the runtime publishes on
 *         prefix: 'my-node',              // namespace for the SVG element ids
 *         defaultWidth: 160               // used when the node sets no debugWidth
 *     });
 *
 * and the runtime publishes `{id, data}` on that topic, `data` being base64
 * image bytes.
 *
 * Load order is not a concern: `<script src>` in a node's HTML loads
 * asynchronously and may arrive after the inline script that calls `attach`, so
 * a call made too early is queued and replayed when this file lands. That is why
 * every node calls `attach` through the shim below rather than reaching for an
 * object that might not exist yet.
 */

(function (global) {
    'use strict';

    // The shim a node's inline script may already have created to queue calls.
    var registry = global.RosepetalDebugImageCanvas
        || (global.RosepetalDebugImageCanvas = {});

    function create(options) {
        var topic = options.topic;
        var prefix = options.prefix;
        var defaultWidth = options.defaultWidth || 160;

        var latestImages = {};
        var imgId = function (id) { return prefix + '-img-' + id; };
        var bubbleId = function (id) { return prefix + '-bubble-' + id; };

        var remove = function (id) {
            var $img = document.getElementById(imgId(id));
            var $bubble = document.getElementById(bubbleId(id));

            $img && $img.remove();
            $bubble && $bubble.remove();
            delete latestImages[id];
        };

        var render = function (id, data, node) {
            var imageWidth = (node && node.debugWidth) || defaultWidth;
            var i = new Image();
            var $img = document.getElementById(imgId(id));

            if (!$img) {
                var $container = document.getElementById(id);
                if (!$container) return;

                var bubble = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
                bubble.setAttribute('id', bubbleId(id));
                bubble.setAttribute('style', 'fill:#E8F0E8');
                bubble.setAttribute('stroke', '#999999');
                bubble.setAttribute('role', 'img');
                bubble.setAttribute('aria-label', 'Debug image for node ' + id);
                $container.insertBefore(bubble, $container.lastChild.nextSibling);

                var img = document.createElementNS('http://www.w3.org/2000/svg', 'image');
                img.setAttribute('id', imgId(id));
                img.setAttribute('x', '0');
                img.setAttribute('y', '45');
                img.setAttribute('width', imageWidth);
                img.setAttribute('role', 'img');
                img.setAttribute('aria-label', 'Debug image output for node ' + id);
                img.setAttribute('tabindex', '0');
                img.addEventListener('click', function () { remove(id); }, { once: true });
                img.addEventListener('keydown', function (e) {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        remove(id);
                    }
                });
                $container.insertBefore(img, $container.lastChild.nextSibling);
                $img = img;
            }

            // The bubble is drawn around the image, so its shape has to wait for
            // the decode rather than assume an aspect ratio.
            i.onload = function () {
                var height = imageWidth * i.height / i.width;
                $img.setAttribute('height', height);

                var $bubble = document.getElementById(bubbleId(id));
                if (!$bubble) return;

                var left = 0;
                var top = 47;
                var right = imageWidth;
                var bottom = 45 + parseInt(height, 10);

                $bubble.setAttribute('points', [
                    (left + 4) + ',' + (top - 17),
                    (left + 4) + ',' + top,
                    right + ',' + top,
                    right + ',' + bottom,
                    left + ',' + bottom,
                    left + ',' + (top - 21)
                ].join(' '));
            };

            // The type comes from the bytes, not from a guess. Every node here
            // publishes JPEG, but this said `image/png` for all of them and got
            // away with it because browsers sniff the content anyway. Reading
            // the first characters of the base64 is cheap and stops it being a
            // lie that happens to work.
            var imageSrc = data.indexOf('http') === 0
                ? data
                : 'data:' + (data.indexOf('/9j/') === 0 ? 'image/jpeg' : 'image/png')
                    + ';base64,' + data;
            $img.setAttribute('href', imageSrc);
            i.src = imageSrc;
        };

        // The canvas is rebuilt on every deploy, so without this the picture
        // would vanish each time the flow is saved.
        var redraw = function (node) {
            if (!node) return;
            var id = node.id;
            var $img = document.getElementById(imgId(id));
            var $bubble = document.getElementById(bubbleId(id));

            $img && $img.remove();
            $bubble && $bubble.remove();

            if (latestImages[id]) {
                render(id, latestImages[id], node);
            }
        };

        RED.events.on('editor:save', redraw);

        RED.comms.subscribe(topic, function (event, data) {
            if (data && data.hasOwnProperty('data')) {
                latestImages[data.id] = data.data;
                render(data.id, data.data, RED.nodes.node(data.id));
            }
        });

        return { remove: remove, redraw: redraw, render: render };
    }

    var pending = registry.pending || [];
    registry.attach = create;
    registry.pending = null;
    pending.forEach(create);
})(window);
