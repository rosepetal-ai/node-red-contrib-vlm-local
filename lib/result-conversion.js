/**
 * The coordinate formats a result can be said in, and the maths between them.
 *
 * One contract, for every node in this package and every task: coordinates are
 * relative to the image, 0-1, and a result carries every format that can be
 * derived from what the model produced. A flow picks the one its next node
 * wants — `2points` for a crop, `raw_boxes` for a drawing — without the panel
 * having to be told in advance and without a redeploy when it changes its
 * mind.
 *
 * It used to be a choice, and the choice was destructive: the formats you had
 * not ticked were deleted, so asking for `raw_boxes` on a segmentation model
 * threw away the polygons, ticking nothing left the item with no geometry, and
 * the `triton-inferencer`, which never ran this at all, answered the same model
 * in a different shape than the `inferencer`.
 *
 * There was also a `mask` format, a raster the size of the image built from the
 * contours. Nothing downstream read it, every item carried its own copy —
 * megabytes a message — and a flow that wants pixels can rasterize the polygons
 * where it needs them. It is gone with the rest of the selector.
 */

'use strict';

const DETECTION_FORMAT_LIST = [
	'raw_boxes',
	'xywhr',
	'cwh',
	'4points',
	'2points'
];

const DETECTION_SOURCE_PRIORITY = [
	'raw_boxes',
	'4points',
	'cwh',
	'xywhr',
	'2points'
];

const toFiniteNumber = (value) => {
	const num = typeof value === 'number' ? value : Number(value);
	return Number.isFinite(num) ? num : 0;
};

const degreesToRadians = (value) => (toFiniteNumber(value) * Math.PI) / 180;
const radiansToDegrees = (value) => (toFiniteNumber(value) * 180) / Math.PI;
const ensureRadians = (value) => toFiniteNumber(value);

const ensureFiniteNumber = (value, field) => {
	const num = typeof value === 'number' ? value : Number(value);
	if (!Number.isFinite(num)) {
		throw new TypeError(`Invalid numeric value for ${field}`);
	}
	return num;
};

const deepClone = (value) => {
	if (Array.isArray(value)) return value.map(deepClone);
	if (value && typeof value === 'object') {
		if (value instanceof Date) return new Date(value);
		if (ArrayBuffer.isView(value)) return value.slice ? value.slice() : new value.constructor(value);
		if (value instanceof ArrayBuffer) return value.slice(0);
		return Object.keys(value).reduce((acc, key) => {
			acc[key] = deepClone(value[key]);
			return acc;
		}, {});
	}
	return value;
};


const looksLikePoint = (candidate) => Array.isArray(candidate) && candidate.length >= 2 && candidate.every(Number.isFinite);

const looksLikeContour = (candidate) => Array.isArray(candidate) && candidate.length > 0 && candidate.every(looksLikePoint);

const extractBoundsFromPolygons = (polygons) => {
	if (!Array.isArray(polygons) || polygons.length === 0) return [];

	let xmin = Infinity;
	let xmax = -Infinity;
	let ymin = Infinity;
	let ymax = -Infinity;

	const processPoints = (points) => {
		points.forEach(([x, y]) => {
			const px = ensureFiniteNumber(x, 'polygon.x');
			const py = ensureFiniteNumber(y, 'polygon.y');
			xmin = Math.min(xmin, px);
			xmax = Math.max(xmax, px);
			ymin = Math.min(ymin, py);
			ymax = Math.max(ymax, py);
		});
	};

	polygons.forEach((entry) => {
		if (!Array.isArray(entry)) return;
		if (looksLikeContour(entry)) {
			processPoints(entry);
		} else {
			entry.forEach((contour) => {
				if (looksLikeContour(contour)) {
					processPoints(contour);
				}
			});
		}
	});

	if (xmin === Infinity) return [];
	return [{ xmin, ymin, xmax, ymax }];
};

const boundsToCanonical = (bounds) => bounds.map(({ xmin, ymin, xmax, ymax }) => {
	const minX = ensureFiniteNumber(xmin, 'xmin');
	const minY = ensureFiniteNumber(ymin, 'ymin');
	const maxX = ensureFiniteNumber(xmax, 'xmax');
	const maxY = ensureFiniteNumber(ymax, 'ymax');

	const width = Math.max(0, maxX - minX);
	const height = Math.max(0, maxY - minY);

	const cx = minX + width / 2;
	const cy = minY + height / 2;

	return { cx, cy, w: width, h: height, r: 0 };
});

const pointsToCanonical = (points) => {
	if (!Array.isArray(points) || points.length !== 4) {
		throw new TypeError('raw_boxes and 4points require exactly 4 points');
	}

	const cleanPoints = points.map((point, index) => {
		if (!looksLikePoint(point)) {
			throw new TypeError(`Invalid point at index ${index}`);
		}
		const [x, y] = point;
		return [ensureFiniteNumber(x, `point${index}.x`), ensureFiniteNumber(y, `point${index}.y`)];
	});

	const [p0, p1, p2, p3] = cleanPoints;
	const cx = (p0[0] + p1[0] + p2[0] + p3[0]) / 4;
	const cy = (p0[1] + p1[1] + p2[1] + p3[1]) / 4;

	const edgeA = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]); // Top edge = width
	const edgeB = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]); // Right edge = height

	const width = edgeA;
	const height = edgeB;
	const rotation = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]);

	return { cx, cy, w: width, h: height, r: rotation };
};

const ensureArrayOfPointBoxes = (rawBoxes) => {
	if (!Array.isArray(rawBoxes)) {
		throw new TypeError('raw_boxes must be an array');
	}

	if (rawBoxes.length === 0) return [];

	const first = rawBoxes[0];
	if (looksLikePoint(first)) {
		return [rawBoxes.map(point => point.slice())];
	}

	return rawBoxes.map((box, index) => {
		if (!Array.isArray(box)) {
			throw new TypeError(`raw_boxes entry at index ${index} must be an array`);
		}
		if (box.length !== 4 || !box.every(looksLikePoint)) {
			throw new TypeError('Each raw_boxes entry must contain four points');
		}
		return box.map(point => point.slice());
	});
};

const rawBoxesToCanonical = (rawBoxes) => ensureArrayOfPointBoxes(rawBoxes).map(pointsToCanonical);

const fourPointsToCanonical = (value) => {
	const normalizeObjectPoints = (obj) => {
		const points = [];
		for (let i = 0; i < 4; i += 1) {
			const xKey = `x${i}`;
			const yKey = `y${i}`;
			if (!(xKey in obj) || !(yKey in obj)) {
				throw new TypeError('4points objects must contain x0..x3 and y0..y3');
			}
			points.push([
				ensureFiniteNumber(obj[xKey], xKey),
				ensureFiniteNumber(obj[yKey], yKey)
			]);
		}
		return points;
	};

	const normalize = (candidate) => {
		if (Array.isArray(candidate)) {
			if (candidate.length === 0) return [];
			if (typeof candidate[0] === 'object' && !Array.isArray(candidate[0])) {
				return candidate.map(normalizeObjectPoints);
			}
			return candidate.map(point => {
				if (!looksLikePoint(point)) {
					throw new TypeError('4points array entries must be [x, y] pairs');
				}
				return [ensureFiniteNumber(point[0], '4points.x'), ensureFiniteNumber(point[1], '4points.y')];
			});
		}

		if (candidate && typeof candidate === 'object') {
			return normalizeObjectPoints(candidate);
		}

		throw new TypeError('Invalid 4points entry');
	};

	if (Array.isArray(value)) {
		if (value.length === 0) return [];
		return value.map((entry) => pointsToCanonical(normalize(entry)));
	}

	if (value && typeof value === 'object') {
		return [pointsToCanonical(normalize(value))];
	}

	throw new TypeError('4points must be an array or object');
};

const boxFromXYWH = (entry, isCenter) => {
	const x = ensureFiniteNumber(entry.x, 'x');
	const y = ensureFiniteNumber(entry.y, 'y');
	const w = Math.max(0, ensureFiniteNumber(entry.w, 'w'));
	const h = Math.max(0, ensureFiniteNumber(entry.h, 'h'));
	const r = degreesToRadians(entry.r);

	if (isCenter) {
		return { cx: x, cy: y, w, h, r };
	}

	return { cx: x + w / 2, cy: y + h / 2, w, h, r };
};

const xywhrToCanonical = (value) => {
	const normalize = (candidate) => {
		if (!candidate || typeof candidate !== 'object') {
			throw new TypeError('xywhr entries must be objects');
		}
		return boxFromXYWH(candidate, false);
	};

	if (Array.isArray(value)) {
		return value.map(normalize);
	}

	return [normalize(value)];
};

const cwhToCanonical = (value) => {
	const normalize = (candidate) => {
		if (!candidate || typeof candidate !== 'object') {
			throw new TypeError('cwh entries must be objects');
		}
		return boxFromXYWH(candidate, true);
	};

	if (Array.isArray(value)) {
		return value.map(normalize);
	}

	return [normalize(value)];
};

const twoPointsToCanonical = (value) => {
	const normalize = (candidate) => {
		if (!candidate || typeof candidate !== 'object') {
			throw new TypeError('2points entries must be objects');
		}
		const xmin = ensureFiniteNumber(candidate.xmin, 'xmin');
		const ymin = ensureFiniteNumber(candidate.ymin, 'ymin');
		const xmax = ensureFiniteNumber(candidate.xmax, 'xmax');
		const ymax = ensureFiniteNumber(candidate.ymax, 'ymax');
		const width = Math.max(0, xmax - xmin);
		const height = Math.max(0, ymax - ymin);
		return { cx: xmin + width / 2, cy: ymin + height / 2, w: width, h: height, r: 0 };
	};

	if (Array.isArray(value)) {
		return value.map(normalize);
	}

	return [normalize(value)];
};

const canonicalToRawBoxes = (boxes) => boxes.map((box) => canonicalToPoints(box));

const canonicalToPoints = (box) => {
	const cx = ensureFiniteNumber(box.cx, 'cx');
	const cy = ensureFiniteNumber(box.cy, 'cy');
	const w = Math.max(0, ensureFiniteNumber(box.w, 'w'));
	const h = Math.max(0, ensureFiniteNumber(box.h, 'h'));
	const r = ensureRadians(box.r);

	const halfW = w / 2;
	const halfH = h / 2;

	const cos = Math.cos(r);
	const sin = Math.sin(r);

	const rotate = (x, y) => [
		cx + x * cos - y * sin,
		cy + x * sin + y * cos
	];

	return [
		rotate(-halfW, -halfH),
		rotate(halfW, -halfH),
		rotate(halfW, halfH),
		rotate(-halfW, halfH)
	];
};

const canonicalToXYWHR = (boxes) => boxes.map((box) => {
	const cx = ensureFiniteNumber(box.cx, 'cx');
	const cy = ensureFiniteNumber(box.cy, 'cy');
	const w = Math.max(0, ensureFiniteNumber(box.w, 'w'));
	const h = Math.max(0, ensureFiniteNumber(box.h, 'h'));
	const r = radiansToDegrees(box.r);
	return { x: cx - w / 2, y: cy - h / 2, w, h, r };
});

const canonicalToCWH = (boxes) => boxes.map((box) => ({
	x: ensureFiniteNumber(box.cx, 'cx'),
	y: ensureFiniteNumber(box.cy, 'cy'),
	w: Math.max(0, ensureFiniteNumber(box.w, 'w')),
	h: Math.max(0, ensureFiniteNumber(box.h, 'h')),
	r: radiansToDegrees(box.r)
}));

const canonicalToFourPoints = (boxes) => canonicalToRawBoxes(boxes).map((points) => ({
	x0: points[0][0],
	y0: points[0][1],
	x1: points[1][0],
	y1: points[1][1],
	x2: points[2][0],
	y2: points[2][1],
	x3: points[3][0],
	y3: points[3][1]
}));

const canonicalToTwoPoints = (boxes) => boxes.map((box) => {
	const cx = ensureFiniteNumber(box.cx, 'cx');
	const cy = ensureFiniteNumber(box.cy, 'cy');
	const w = Math.max(0, ensureFiniteNumber(box.w, 'w'));
	const h = Math.max(0, ensureFiniteNumber(box.h, 'h'));
	return {
		xmin: cx - w / 2,
		ymin: cy - h / 2,
		xmax: cx + w / 2,
		ymax: cy + h / 2
	};
});

const DETECTION_FROM_FORMAT = {
	raw_boxes: rawBoxesToCanonical,
	'4points': fourPointsToCanonical,
	cwh: cwhToCanonical,
	xywhr: xywhrToCanonical,
	'2points': twoPointsToCanonical
};

const DETECTION_TO_FORMAT = {
	raw_boxes: canonicalToRawBoxes,
	'4points': canonicalToFourPoints,
	cwh: canonicalToCWH,
	xywhr: canonicalToXYWHR,
	'2points': canonicalToTwoPoints
};

const findDetectionCanonical = (result) => {
	for (const format of DETECTION_SOURCE_PRIORITY) {
		if (result[format] !== undefined && result[format] !== null) {
			const boxes = DETECTION_FROM_FORMAT[format](result[format]);
			return { boxes, sourceFormat: format };
		}
	}
	return null;
};

/** The bounding box of a segmentation or anomaly item, from its contours. */
const findSegmentationDerivedDetections = (result) => {
	if (result.polygons === undefined || result.polygons === null) return null;
	const bounds = extractBoundsFromPolygons(result.polygons);
	if (bounds.length === 0) return { boxes: [], sourceFormat: 'polygons' };
	return { boxes: boundsToCanonical(bounds), sourceFormat: 'polygons' };
};

const buildDetectionOutputs = (canonicalBoxes, requestedFormats) => {
	const outputs = {};
	requestedFormats.forEach((format) => {
		const formatter = DETECTION_TO_FORMAT[format];
		const result = formatter(canonicalBoxes);
		// If we have exactly one box, unwrap it from the array for cleaner output
		if (result.length === 1) {
			outputs[format] = result[0];
		} else {
			outputs[format] = result;
		}
	});
	return outputs;
};

// ---------------------------------------------------------------------------
// Expansion: every coordinate format, always
// ---------------------------------------------------------------------------

/** The geometry formats a result carries. */
const COORDINATE_FORMATS = [...DETECTION_FORMAT_LIST, 'polygons'];

/**
 * Round a derived geometry blob to 6 decimals.
 *
 * Deriving a box goes through a centre and a half-extent, so a corner that
 * was exactly 0.1 comes back as 0.10000000000000003. At 0-1 with 6 decimals
 * the residual is sub-pixel for any image under a million pixels a side, and
 * the payload stops being full of noise. The C++ backend writes its own
 * coordinates at the same precision.
 */
const roundGeometry = (value) => {
	if (typeof value === 'number') {
		return Number.isFinite(value) ? Math.round(value * 1e6) / 1e6 : value;
	}
	if (Array.isArray(value)) return value.map(roundGeometry);
	if (value && typeof value === 'object') {
		return Object.keys(value).reduce((acc, key) => {
			acc[key] = roundGeometry(value[key]);
			return acc;
		}, {});
	}
	return value;
};

/**
 * Add every derivable coordinate format to one result item, keeping what is
 * already there.
 *
 * This replaces the old "pick your formats" transform, which *removed* the
 * formats you had not ticked: selecting `raw_boxes` on a segmentation or
 * anomaly model threw the polygons away, and selecting nothing at all left the
 * item with no geometry whatsoever. Since the coordinates are normalized 0-1,
 * carrying all five box shapes plus the polygons costs a few dozen numbers,
 * which is not worth a setting.
 *
 * Boxes are derived from whichever source the item actually has: a real box
 * format if there is one, otherwise the bounding box of the polygons.
 */
const expandSingleResult = (result) => {
	if (!result || typeof result !== 'object' || Array.isArray(result)) {
		throw new TypeError('Individual result must be an object containing format data');
	}

	const output = {};
	Object.keys(result).forEach((key) => {
		output[key] = deepClone(result[key]);
	});

	let detectionData = findDetectionCanonical(result);
	if (!detectionData) detectionData = findSegmentationDerivedDetections(result);

	if (detectionData) {
		const missing = DETECTION_FORMAT_LIST.filter((format) => {
			const value = output[format];
			return value === undefined || value === null;
		});
		if (missing.length > 0) {
			Object.assign(output, roundGeometry(buildDetectionOutputs(detectionData.boxes, missing)));
		}
	}

	return output;
};

/**
 * `expandSingleResult` over one image's result, whatever shape it has: the
 * list of items a detection or segmentation returns, the single object an
 * anomaly or classification returns, or an empty list.
 */
const expandResultFormats = (result) => {
	if (Array.isArray(result)) {
		return result.map((item) => (item && typeof item === 'object' && !Array.isArray(item)
			? expandSingleResult(item)
			: item));
	}
	if (!result || typeof result !== 'object') return result;
	// Classification, OCR and the rest have no geometry; expanding them is a
	// no-op that would still deep-clone, so let them through untouched.
	const hasGeometry = COORDINATE_FORMATS.some((format) => result[format] !== undefined && result[format] !== null);
	if (!hasGeometry) return result;
	return expandSingleResult(result);
};

module.exports = { expandResultFormats, COORDINATE_FORMATS };
