"""
Utilities for interfacing with the
`ETA library <https://github.com/voxel51/eta>`_.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from collections import defaultdict
import warnings
import numpy as np
import eta.core.data as etad
import eta.core.events as etae
import eta.core.frames as etaf
import eta.core.frameutils as etafu
import eta.core.geometry as etag
import eta.core.image as etai
import eta.core.keypoints as etak
import eta.core.learning as etal
import eta.core.objects as etao
import eta.core.polylines as etap
import eta.core.utils as etau
import eta.core.video as etav
import fiftyone.core.labels as fol
import fiftyone.core.models as fom
_IMAGE_MODELS = (etal.ImageModel, etal.ImageClassifier, etal.ObjectDetector, etal.ImageSemanticSegmenter)
_VIDEO_MODELS = (etal.VideoModel, etal.VideoClassifier, etal.VideoObjectDetector, etal.VideoEventDetector, etal.VideoSemanticSegmenter)

class ETAModelConfig(fom.ModelConfig):
    """Meta-config class that encapsulates the configuration of an
    `eta.core.learning.Model` that is to be run via the :class:`ETAModel`
    wrapper.

    Example::

        import fiftyone.core.models as fom

        model = fom.load_model({
            "type": "fiftyone.utils.eta.ETAModel",
            "config": {
                "type": "eta.detectors.YOLODetector",
                "config": {
                    "model_name": "yolo-v2-coco"
                }
            }
        })

    Args:
        type: the fully-qualified class name of the
            :class:`fiftyone.core.models.Model` subclass, which must be
            :class:`ETAModel` or a subclass of it
        config: a dict containing the ``eta.core.learning.ModelConfig`` for the
            ETA model
    """

    @property
    def confidence_thresh(self):
        if False:
            i = 10
            return i + 15
        'The confidence threshold of the underlying ``eta.core.model.Model``.\n\n        Note that this may not be defined for some models.\n        '
        return self.config.confidence_thresh

    @confidence_thresh.setter
    def confidence_thresh(self, confidence_thresh):
        if False:
            print('Hello World!')
        self.config.confidence_thresh = confidence_thresh

class ETAModel(fom.Model, fom.EmbeddingsMixin, fom.LogitsMixin):
    """Wrapper for running an ``eta.core.learning.Model`` model.

    Args:
        config: an :class:`ETAModelConfig`
    """

    def __init__(self, config, _model=None):
        if False:
            for i in range(10):
                print('nop')
        if _model is None:
            _model = config.build()
        self.config = config
        self._model = _model
        fom.LogitsMixin.__init__(self)

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self._model.__enter__()
        return self

    def __exit__(self, *args):
        if False:
            print('Hello World!')
        self._model.__exit__(*args)

    @property
    def media_type(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self._model, _IMAGE_MODELS):
            return 'image'
        if isinstance(self._model, _VIDEO_MODELS):
            return 'video'
        return None

    @property
    def ragged_batches(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._model.ragged_batches
        except AttributeError:
            return True

    @property
    def transforms(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._model.transforms
        except AttributeError:
            return None

    @property
    def preprocess(self):
        if False:
            return 10
        try:
            return self._model.preprocess
        except AttributeError:
            return False

    @preprocess.setter
    def preprocess(self, value):
        if False:
            while True:
                i = 10
        try:
            self._model.preprocess = value
        except AttributeError:
            pass

    @property
    def has_logits(self):
        if False:
            return 10
        return isinstance(self._model, etal.ExposesProbabilities) and isinstance(self._model, etal.Classifier) and self._model.exposes_probabilities

    @property
    def has_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(self._model, etal.ExposesFeatures) and isinstance(self._model, etal.Classifier) and self._model.exposes_features

    def _ensure_embeddings(self):
        if False:
            print('Hello World!')
        if not self.has_embeddings:
            raise ValueError('This model instance does not expose embeddings')

    def get_embeddings(self):
        if False:
            i = 10
            return i + 15
        self._ensure_embeddings()
        embeddings = self._model.get_features()
        embeddings = _squeeze_extra_unit_dims(embeddings)
        return embeddings.astype(float, copy=False)

    def embed(self, arg):
        if False:
            print('Hello World!')
        self._ensure_embeddings()
        self.predict(arg)
        return self.get_embeddings()

    def embed_all(self, args):
        if False:
            return 10
        self._ensure_embeddings()
        if isinstance(self._model, etal.ImageClassifier):
            self._model.predict_all(args)
            return self.get_embeddings()
        if isinstance(self._model, etal.ObjectDetector):
            self._model.detect_all(args)
            return self.get_embeddings()
        if isinstance(self._model, etal.ImageSemanticSegmenter):
            self._model.segment_all(args)
            return self.get_embeddings()
        return np.concatenate(tuple((self.embed(arg) for arg in args)))

    def predict(self, arg):
        if False:
            i = 10
            return i + 15
        if isinstance(self._model, etal.ImageClassifier):
            eta_labels = self._model.predict(arg)
        elif isinstance(self._model, etal.VideoFramesClassifier):
            eta_labels = self._model.predict(arg)
        elif isinstance(self._model, etal.VideoClassifier):
            eta_labels = self._model.predict(arg)
        elif isinstance(self._model, etal.Classifier):
            eta_labels = self._model.predict(arg)
        elif isinstance(self._model, etal.ObjectDetector):
            eta_labels = self._model.detect(arg)
        elif isinstance(self._model, etal.VideoFramesObjectDetector):
            eta_labels = self._model.detect(arg)
        elif isinstance(self._model, etal.VideoObjectDetector):
            eta_labels = self._model.detect(arg)
        elif isinstance(self._model, etal.Detector):
            eta_labels = self._model.detect(arg)
        elif isinstance(self._model, etal.ImageSemanticSegmenter):
            eta_labels = self._model.segment(arg)
        elif isinstance(self._model, etal.VideoSemanticSegmenter):
            eta_labels = self._model.segment(arg)
        elif isinstance(self._model, etal.SemanticSegmenter):
            eta_labels = self._model.segment(arg)
        elif isinstance(self._model, etal.ImageModel):
            eta_labels = self._model.process(arg)
        elif isinstance(self._model, etal.VideoModel):
            eta_labels = self._model.process(arg)
        else:
            raise ValueError("Unsupported model type '%s'" % self._model.__class__)
        eta_labels = self._parse_predictions(eta_labels)
        label = _from_eta_labels(eta_labels)
        if self.has_logits and self.store_logits:
            logits = np.log(self._model.get_probabilities()[0])
            _add_logits(label, logits)
        return label

    def predict_all(self, args):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self._model, etal.ImageClassifier):
            eta_labels_batch = self._model.predict_all(args)
        elif isinstance(self._model, etal.ObjectDetector):
            eta_labels_batch = self._model.detect_all(args)
        elif isinstance(self._model, etal.ImageSemanticSegmenter):
            eta_labels_batch = self._model.segment_all(args)
        else:
            return [self.predict(arg) for arg in args]
        eta_labels_batch = self._parse_predictions(eta_labels_batch)
        labels = [_from_eta_labels(el) for el in eta_labels_batch]
        if self.has_logits and self.store_logits:
            logits = np.log(self._model.get_probabilities())
            for (label, _logits) in zip(labels, logits):
                _add_logits(label, _logits)
        return labels

    def _parse_predictions(self, eta_labels_or_batch):
        if False:
            return 10
        if not isinstance(self._model, etal.Classifier) or self._model.is_multilabel:
            return eta_labels_or_batch
        if isinstance(eta_labels_or_batch, list):
            return [attrs[0] if attrs else None for attrs in eta_labels_or_batch]
        if not eta_labels_or_batch:
            return None
        return eta_labels_or_batch[0]

    @classmethod
    def from_eta_model(cls, model):
        if False:
            for i in range(10):
                print('nop')
        'Builds an :class:`ETAModel` for running the provided\n        ``eta.core.learning.Model`` instance.\n\n        Args:\n            model: an ``eta.core.learning.Model`` instance\n\n        Returns:\n            an :class:`ETAModel`\n        '
        return cls(model.config, _model=model)

def from_image_labels(image_labels_or_path, prefix=None, labels_dict=None, multilabel=False, skip_non_categorical=False):
    if False:
        i = 10
        return i + 15
    'Loads the ``eta.core.image.ImageLabels`` or\n    ``eta.core.frames.FrameLabels`` into a dictionary of labels.\n\n    Provide ``labels_dict`` if you want to customize which components of\n    the labels are expanded. Otherwise, all labels are expanded as\n    explained below.\n\n    If ``multilabel`` is False, frame attributes will be stored in separate\n    :class:`Classification` fields with names ``prefix + attr.name``.\n\n    If ``multilabel`` if True, all frame attributes will be stored in a\n    :class:`Classifications` field called ``prefix + "attributes"``.\n\n    Objects are expanded into fields with names ``prefix + obj.name``, or\n    ``prefix + "detections"`` for objects that do not have their ``name``\n    field populated.\n\n    Polylines are expanded into fields with names\n    ``prefix + polyline.name``, or ``prefix + "polylines"`` for polylines\n    that do not have their ``name`` field populated.\n\n    Keypoints are expanded into fields with names\n    ``prefix + keypoints.name``, or ``prefix + "keypoints"`` for keypoints\n    that do not have their ``name`` field populated.\n\n    Segmentation masks are expanded into a field with name ``prefix + "mask"``.\n\n    Args:\n        image_labels_or_path: can be a ``eta.core.image.ImageLabels`` instance,\n            a ``eta.core.frames.FrameLabels`` instance, a serialized dict\n            representation of either, or the path to either on disk\n        prefix (None): a string prefix to prepend to each field name in the\n            output dict\n        labels_dict (None): a dictionary mapping names of labels to keys to\n            assign them in the output dictionary\n        multilabel (False): whether to store attributes in a single\n            :class:`Classifications` instance\n        skip_non_categorical (False): whether to skip non-categorical\n            attributes (True) or cast them to strings (False)\n\n    Returns:\n        a dict mapping names to :class:`fiftyone.core.labels.Label` instances\n    '
    if etau.is_str(image_labels_or_path):
        frame_labels = etaf.FrameLabels.from_json(image_labels_or_path)
    elif isinstance(image_labels_or_path, dict):
        frame_labels = etaf.FrameLabels.from_dict(image_labels_or_path)
    else:
        frame_labels = image_labels_or_path
    if frame_labels is None:
        return None
    if labels_dict is not None:
        return _expand_with_labels_dict(frame_labels, labels_dict, multilabel, skip_non_categorical)
    return _expand_with_prefix(frame_labels, prefix, multilabel, skip_non_categorical)

def to_image_labels(labels, warn_unsupported=True):
    if False:
        while True:
            i = 10
    'Converts the image label(s) to ``eta.core.image.ImageLabels`` format.\n\n    Args:\n        labels: a :class:`fiftyone.core.labels.Label` instance or a dict\n            mapping names to :class:`fiftyone.core.labels.Label` instances\n        warn_unsupported (True): whether to issue warnings if unsupported label\n            values are encountered\n\n    Returns:\n        an ``eta.core.image.ImageLabels`` instance\n    '
    image_labels = etai.ImageLabels()
    if labels is None:
        return image_labels
    if not isinstance(labels, dict):
        labels = {'labels': labels}
    _add_frame_labels(image_labels, labels, warn_unsupported=warn_unsupported)
    return image_labels

def from_video_labels(video_labels_or_path, prefix=None, labels_dict=None, frame_labels_dict=None, multilabel=False, skip_non_categorical=False):
    if False:
        for i in range(10):
            print('nop')
    'Loads the ``eta.core.video.VideoLabels`` into a frame labels dictionary.\n\n    Args:\n        video_labels_or_path: can be a ``eta.core.video.VideoLabels`` instance,\n            a serialized dict representation of one, or the path to one on disk\n        prefix (None): a string prefix to prepend to each label name in the\n            expanded sample/frame label dictionaries\n        labels_dict (None): a dictionary mapping names of attributes/objects\n            in the sample labels to field names into which to expand them. By\n            default, all sample labels are loaded\n        frame_labels_dict (None): a dictionary mapping names of\n            attributes/objects in the frame labels to field names into which to\n            expand them. By default, all frame labels are loaded\n        multilabel (False): whether to store attributes in a single\n            :class:`fiftyone.core.labels.Classifications` instance\n        skip_non_categorical (False): whether to skip non-categorical\n            attributes (True) or cast them to strings (False)\n\n    Returns:\n        a tuple of\n\n        -   **label**: a dict mapping sample field names to\n            :class:`fiftyone.core.labels.Label` instances\n        -   **frames**: a dict mapping frame numbers to dicts that map label\n            fields to :class:`fiftyone.core.labels.Label` instances\n    '
    if etau.is_str(video_labels_or_path):
        video_labels = etav.VideoLabels.from_json(video_labels_or_path)
    elif isinstance(video_labels_or_path, dict):
        video_labels = etav.VideoLabels.from_dict(video_labels_or_path)
    else:
        video_labels = video_labels_or_path
    if video_labels is None:
        return (None, None)
    if (video_labels.has_video_attributes or video_labels.has_video_events) and (labels_dict is None or labels_dict):
        if labels_dict is not None:
            label = _expand_with_labels_dict(video_labels, labels_dict, multilabel, skip_non_categorical)
        else:
            label = _expand_with_prefix(video_labels, prefix, multilabel, skip_non_categorical)
    else:
        label = None
    if video_labels.frames and (frame_labels_dict is None or frame_labels_dict):
        frames = {}
        for frame_number in video_labels:
            frames[frame_number] = from_image_labels(video_labels[frame_number], prefix=prefix, labels_dict=frame_labels_dict, multilabel=multilabel, skip_non_categorical=skip_non_categorical)
    else:
        frames = None
    return (label, frames)

def to_video_labels(label=None, frames=None, support=None, warn_unsupported=True):
    if False:
        while True:
            i = 10
    'Converts the given labels to ``eta.core.video.VideoLabels`` format.\n\n    Args:\n        label (None): video-level labels provided as a\n            :class:`fiftyone.core.labels.Label` instance or dict mapping field\n            names to :class:`fiftyone.core.labels.Label` instances\n        frames (None): frame-level labels provided as a dict mapping frame\n            numbers to dicts mapping field names to\n            :class:`fiftyone.core.labels.Label` instances\n        support (None): an optional ``[first, last]`` support to store on the\n            returned labels\n        warn_unsupported (True): whether to issue warnings if unsupported label\n            values are encountered\n\n    Returns:\n        a ``eta.core.video.VideoLabels``\n    '
    if support is not None:
        support = etafu.FrameRanges.build_simple(*support)
    video_labels = etav.VideoLabels(support=support)
    if label is not None:
        if not isinstance(label, dict):
            label = {'labels': label}
        _add_video_labels(video_labels, label, warn_unsupported=warn_unsupported)
    if frames is not None:
        for (frame_number, frame) in frames.items():
            frame_labels = etav.VideoFrameLabels(frame_number)
            _add_frame_labels(frame_labels, frame, warn_unsupported=warn_unsupported)
            video_labels[frame_number] = frame_labels
    return video_labels

def to_attribute(classification, name=None):
    if False:
        while True:
            i = 10
    'Returns an ``eta.core.data.Attribute`` representation of the\n    :class:`fiftyone.core.labels.Classification`.\n\n    Args:\n        classification: a :class:`fiftyone.core.labels.Classification`\n        name (None): the name of the label field\n\n    Returns:\n        a ``eta.core.data.CategoricalAttribute``\n    '
    return etad.CategoricalAttribute(name, classification.label, confidence=classification.confidence, tags=classification.tags)

def from_attribute(attr):
    if False:
        for i in range(10):
            print('nop')
    'Creates a :class:`fiftyone.core.labels.Classification` from an\n    ``eta.core.data.Attribute``.\n\n    The attribute value is cast to a string, if necessary.\n\n    Args:\n        attr: an ``eta.core.data.Attribute``\n\n    Returns:\n        a :class:`fiftyone.core.labels.Classification`\n    '
    classification = fol.Classification(label=str(attr.value))
    try:
        classification.confidence = attr.confidence
    except:
        pass
    return classification

def from_attributes(attrs, skip_non_categorical=False):
    if False:
        for i in range(10):
            print('nop')
    'Creates a :class:`fiftyone.core.labels.Classifications` from a list of\n    attributes.\n\n    Args:\n        attrs: an iterable of ``eta.core.data.Attribute`` instances\n        skip_non_categorical (False): whether to skip non-categorical\n            attributes (True) or cast all attribute values to strings\n            (False)\n\n    Returns:\n        a :class:`fiftyone.core.labels.Classifications`\n    '
    classifications = []
    for attr in attrs:
        if skip_non_categorical and (not etau.is_str(attr.value)):
            continue
        classifications.append(from_attribute(attr))
    return fol.Classifications(classifications=classifications)

def to_detected_object(detection, name=None, extra_attrs=True):
    if False:
        i = 10
        return i + 15
    'Returns an ``eta.core.objects.DetectedObject`` representation of the\n    given :class:`fiftyone.core.labels.Detection`.\n\n    Args:\n        detection: a :class:`fiftyone.core.labels.Detection`\n        name (None): the name of the label field\n        extra_attrs (True): whether to include custom attributes in the\n            conversion\n\n    Returns:\n        an ``eta.core.objects.DetectedObject``\n    '
    label = detection.label
    index = detection.index
    (tlx, tly, w, h) = detection.bounding_box
    brx = tlx + w
    bry = tly + h
    bounding_box = etag.BoundingBox.from_coords(tlx, tly, brx, bry)
    mask = detection.mask
    confidence = detection.confidence
    attrs = _to_eta_attributes(detection, extra_attrs=extra_attrs)
    return etao.DetectedObject(label=label, index=index, bounding_box=bounding_box, mask=mask, confidence=confidence, name=name, attrs=attrs, tags=detection.tags)

def from_detected_object(dobj):
    if False:
        i = 10
        return i + 15
    'Creates a :class:`fiftyone.core.labels.Detection` from an\n    ``eta.core.objects.DetectedObject``.\n\n    Args:\n        dobj: a ``eta.core.objects.DetectedObject``\n\n    Returns:\n        a :class:`fiftyone.core.labels.Detection`\n    '
    (xtl, ytl, xbr, ybr) = dobj.bounding_box.to_coords()
    bounding_box = [xtl, ytl, xbr - xtl, ybr - ytl]
    attributes = _from_eta_attributes(dobj.attrs)
    return fol.Detection(label=dobj.label, bounding_box=bounding_box, confidence=dobj.confidence, index=dobj.index, mask=dobj.mask, tags=dobj.tags, **attributes)

def from_detected_objects(objects):
    if False:
        while True:
            i = 10
    'Creates a :class:`fiftyone.core.labels.Detections` from an\n    ``eta.core.objects.DetectedObjectContainer``.\n\n    Args:\n        objects: a ``eta.core.objects.DetectedObjectContainer``\n\n    Returns:\n        a :class:`fiftyone.core.labels.Detections`\n    '
    return fol.Detections(detections=[from_detected_object(dobj) for dobj in objects])

def to_polyline(polyline, name=None, extra_attrs=True):
    if False:
        for i in range(10):
            print('nop')
    'Returns an ``eta.core.polylines.Polyline`` representation of the given\n    :class:`fiftyone.core.labels.Polyline`.\n\n    Args:\n        polyline: a :class:`fiftyone.core.labels.Polyline`\n        name (None): the name of the label field\n        extra_attrs (True): whether to include custom attributes in the\n            conversion\n\n    Returns:\n        an ``eta.core.polylines.Polyline``\n    '
    attrs = _to_eta_attributes(polyline, extra_attrs=extra_attrs)
    return etap.Polyline(label=polyline.label, confidence=polyline.confidence, index=polyline.index, name=name, points=polyline.points, closed=polyline.closed, filled=polyline.filled, attrs=attrs, tags=polyline.tags)

def from_polyline(polyline):
    if False:
        i = 10
        return i + 15
    'Creates a :class:`fiftyone.core.labels.Polyline` from an\n    ``eta.core.polylines.Polyline``.\n\n    Args:\n        polyline: an ``eta.core.polylines.Polyline``\n\n    Returns:\n        a :class:`fiftyone.core.labels.Polyline`\n    '
    attributes = _from_eta_attributes(polyline.attrs)
    return fol.Polyline(label=polyline.label, points=polyline.points, confidence=polyline.confidence, index=polyline.index, closed=polyline.closed, filled=polyline.filled, tags=polyline.tags, **attributes)

def from_polylines(polylines):
    if False:
        while True:
            i = 10
    'Creates a :class:`fiftyone.core.labels.Polylines` from an\n    ``eta.core.polylines.PolylineContainer``.\n\n    Args:\n        polylines: an ``eta.core.polylines.PolylineContainer``\n\n    Returns:\n        a :class:`fiftyone.core.labels.Polylines`\n    '
    return fol.Polylines(polylines=[from_polyline(p) for p in polylines])

def to_keypoints(keypoint, name=None, extra_attrs=True):
    if False:
        print('Hello World!')
    'Returns an ``eta.core.keypoints.Keypoints`` representation of the given\n    :class:`fiftyone.core.labels.Keypoint`.\n\n    Args:\n        keypoint: a :class:`fiftyone.core.labels.Keypoint`\n        name (None): the name of the label field\n        extra_attrs (True): whether to include custom attributes in the\n            conversion\n\n    Returns:\n        an ``eta.core.keypoints.Keypoints``\n    '
    attrs = _to_eta_attributes(keypoint, extra_attrs=extra_attrs)
    return etak.Keypoints(name=name, label=keypoint.label, index=keypoint.index, points=keypoint.points, confidence=keypoint.confidence, attrs=attrs, tags=keypoint.tags)

def from_keypoint(keypoints):
    if False:
        while True:
            i = 10
    'Creates a :class:`fiftyone.core.labels.Keypoint` from an\n    ``eta.core.keypoints.Keypoints``.\n\n    Args:\n        keypoints: an ``eta.core.keypoints.Keypoints``\n\n    Returns:\n        a :class:`fiftyone.core.labels.Keypoint`\n    '
    attributes = _from_eta_attributes(keypoints.attrs)
    return fol.Keypoint(label=keypoints.label, points=keypoints.points, confidence=keypoints.confidence, index=keypoints.index, tags=keypoints.tags, **attributes)

def from_keypoints(keypoints):
    if False:
        for i in range(10):
            print('nop')
    'Creates a :class:`fiftyone.core.labels.Keypoints` from an\n    ``eta.core.keypoints.KeypointsContainer``.\n\n    Args:\n        keypoints: an ``eta.core.keypoints.KeypointsContainer``\n\n    Returns:\n        a :class:`fiftyone.core.labels.Keypoints`\n    '
    return fol.Keypoints(keypoints=[from_keypoint(k) for k in keypoints])

def to_video_event(temporal_detection, name=None, extra_attrs=True):
    if False:
        while True:
            i = 10
    'Returns an ``eta.core.events.VideoEvent`` representation of the given\n    :class:`fiftyone.core.labels.TemporalDetection`.\n\n    Args:\n        temporal_detection: a :class:`fiftyone.core.labels.TemporalDetection`\n        name (None): the name of the label field\n        extra_attrs (True): whether to include custom attributes in the\n            conversion\n\n    Returns:\n        an ``eta.core.events.VideoEvent``\n    '
    support = etafu.FrameRanges.build_simple(*temporal_detection.support)
    attrs = _to_eta_attributes(temporal_detection, extra_attrs=extra_attrs)
    return etae.VideoEvent(label=temporal_detection.label, confidence=temporal_detection.confidence, name=name, support=support, attrs=attrs, tags=temporal_detection.tags)

def from_video_event(video_event):
    if False:
        for i in range(10):
            print('nop')
    'Creates a :class:`fiftyone.core.labels.TemporalDetection` from an\n    ``eta.core.events.VideoEvent``.\n\n    Args:\n        video_event: an ``eta.core.events.VideoEvent``\n\n    Returns:\n        a :class:`fiftyone.core.labels.TemporalDetection`\n    '
    if video_event.support:
        support = list(video_event.support.limits)
    else:
        support = None
    attributes = _from_eta_attributes(video_event.attrs)
    return fol.TemporalDetection(label=video_event.label, support=support, confidence=video_event.confidence, tags=video_event.tags, **attributes)

def from_video_events(video_events):
    if False:
        print('Hello World!')
    'Creates a :class:`fiftyone.core.labels.TemporalDetections` from an\n    ``eta.core.events.VideoEventContainer``.\n\n    Args:\n        video_events: an ``eta.core.events.VideoEventContainer``\n\n    Returns:\n        a :class:`fiftyone.core.labels.TemporalDetections`\n    '
    return fol.TemporalDetections(detections=[from_video_event(e) for e in video_events])

def _add_frame_labels(frame_labels, labels, warn_unsupported=True):
    if False:
        for i in range(10):
            print('nop')
    for (name, label) in labels.items():
        if isinstance(label, fol.Classification):
            frame_labels.add_attribute(to_attribute(label, name=name))
        elif isinstance(label, fol.Classifications):
            for classification in label.classifications:
                attr = to_attribute(classification, name=name)
                frame_labels.add_attribute(attr)
        elif isinstance(label, fol.Detection):
            frame_labels.add_object(to_detected_object(label, name=name))
        elif isinstance(label, fol.Detections):
            for detection in label.detections:
                dobj = to_detected_object(detection, name=name)
                frame_labels.add_object(dobj)
        elif isinstance(label, fol.Polyline):
            frame_labels.add_polyline(to_polyline(label, name=name))
        elif isinstance(label, fol.Polylines):
            for polyline in label.polylines:
                poly = to_polyline(polyline, name=name)
                frame_labels.add_polyline(poly)
        elif isinstance(label, fol.Keypoint):
            frame_labels.add_keypoints(to_keypoints(label, name=name))
        elif isinstance(label, fol.Keypoints):
            for keypoint in label.keypoints:
                kp = to_keypoints(keypoint, name=name)
                frame_labels.add_keypoints(kp)
        elif isinstance(label, fol.Segmentation):
            frame_labels.mask = label.get_mask()
            frame_labels.tags.extend(label.tags)
        elif warn_unsupported and label is not None:
            msg = "Ignoring unsupported label type '%s'" % label.__class__
            warnings.warn(msg)
    return frame_labels

def _add_video_labels(video_labels, labels, warn_unsupported=True):
    if False:
        print('Hello World!')
    for (name, label) in labels.items():
        if isinstance(label, fol.Classification):
            video_labels.add_video_attribute(to_attribute(label, name=name))
        elif isinstance(label, fol.Classifications):
            for classification in label.classifications:
                attr = to_attribute(classification, name=name)
                video_labels.add_video_attribute(attr)
        elif isinstance(label, fol.TemporalDetection):
            video_labels.add_event(to_video_event(label, name=name))
        elif isinstance(label, fol.TemporalDetections):
            for detection in label.detections:
                event = to_video_event(detection, name=name)
                video_labels.add_event(event)
        elif warn_unsupported and label is not None:
            msg = "Ignoring unsupported label type '%s'" % label.__class__
            warnings.warn(msg)

def _from_eta_labels(eta_labels):
    if False:
        print('Hello World!')
    if isinstance(eta_labels, etad.AttributeContainer):
        label = from_attributes(eta_labels)
    elif isinstance(eta_labels, etad.Attribute):
        label = from_attribute(eta_labels)
    elif isinstance(eta_labels, etao.DetectedObjectContainer):
        label = from_detected_objects(eta_labels)
    elif isinstance(eta_labels, etao.DetectedObject):
        label = from_detected_object(eta_labels)
    elif isinstance(eta_labels, etap.PolylineContainer):
        label = from_polylines(eta_labels)
    elif isinstance(eta_labels, etap.Polyline):
        label = from_polyline(eta_labels)
    elif isinstance(eta_labels, etak.KeypointsContainer):
        label = from_keypoints(eta_labels)
    elif isinstance(eta_labels, etak.Keypoints):
        label = from_keypoint(eta_labels)
    elif isinstance(eta_labels, etae.VideoEventContainer):
        label = from_video_events(eta_labels)
    elif isinstance(eta_labels, etae.VideoEvent):
        label = from_video_event(eta_labels)
    elif isinstance(eta_labels, etav.VideoLabels):
        (_, label) = from_video_labels(eta_labels)
    elif isinstance(eta_labels, etaf.FrameLabels):
        label = from_image_labels(eta_labels)
    elif isinstance(eta_labels, np.ndarray):
        label = fol.Segmentation(mask=eta_labels)
    elif eta_labels is None:
        label = None
    else:
        raise ValueError("Unsupported ETA label type '%s'" % eta_labels.__class__)
    return label

def _from_eta_attributes(attrs):
    if False:
        return 10
    return {a.name: a.value for a in attrs}

def _to_eta_attributes(label, extra_attrs=True, warn_unsupported=True):
    if False:
        i = 10
        return i + 15
    attrs = etad.AttributeContainer()
    if not extra_attrs:
        return attrs
    for (name, value) in label.iter_attributes():
        if etau.is_str(value):
            attrs.add(etad.CategoricalAttribute(name, value))
        elif etau.is_numeric(value):
            attrs.add(etad.NumericAttribute(name, value))
        elif isinstance(value, bool):
            attrs.add(etad.BooleanAttribute(name, value))
        elif warn_unsupported and value is not None:
            msg = "Ignoring unsupported attribute type '%s'" % type(value)
            warnings.warn(msg)
    return attrs

def _expand_with_prefix(video_or_frame_labels, prefix, multilabel, skip_non_categorical):
    if False:
        while True:
            i = 10
    if prefix is None:
        prefix = ''
    labels = {}
    if multilabel:
        labels[prefix + 'attributes'] = from_attributes(video_or_frame_labels.attrs, skip_non_categorical=skip_non_categorical)
    else:
        for attr in video_or_frame_labels.attrs:
            if skip_non_categorical and (not etau.is_str(attr.value)):
                continue
            labels[prefix + attr.name] = from_attribute(attr)
    if isinstance(video_or_frame_labels, etav.VideoLabels):
        events_map = defaultdict(etae.VideoEventContainer)
        for event in video_or_frame_labels.events:
            events_map[prefix + (event.name or 'events')].add(event)
        for (name, events) in events_map.items():
            labels[name] = from_video_events(events)
        return labels
    objects_map = defaultdict(etao.DetectedObjectContainer)
    for dobj in video_or_frame_labels.objects:
        objects_map[prefix + (dobj.name or 'detections')].add(dobj)
    for (name, objects) in objects_map.items():
        labels[name] = from_detected_objects(objects)
    polylines_map = defaultdict(etap.PolylineContainer)
    for polyline in video_or_frame_labels.polylines:
        polylines_map[prefix + (polyline.name or 'polylines')].add(polyline)
    for (name, polylines) in polylines_map.items():
        labels[name] = from_polylines(polylines)
    keypoints_map = defaultdict(etak.KeypointsContainer)
    for keypoints in video_or_frame_labels.keypoints:
        keypoints_map[prefix + (keypoints.name or 'keypoints')].add(keypoints)
    for (name, keypoints) in keypoints_map.items():
        labels[name] = from_keypoints(keypoints)
    if video_or_frame_labels.has_mask:
        labels[prefix + 'mask'] = fol.Segmentation(mask=video_or_frame_labels.mask)
    return labels

def _expand_with_labels_dict(video_or_frame_labels, labels_dict, multilabel, skip_non_categorical):
    if False:
        while True:
            i = 10
    labels = {}
    if multilabel:
        attrs_map = defaultdict(etad.AttributeContainer)
        for attr in video_or_frame_labels.attrs:
            if attr.name not in labels_dict:
                continue
            attrs_map[labels_dict[attr.name]].add(attr)
        for (name, attrs) in attrs_map.items():
            labels[name] = from_attributes(attrs, skip_non_categorical=skip_non_categorical)
    else:
        for attr in video_or_frame_labels.attrs:
            if skip_non_categorical and (not etau.is_str(attr.value)):
                continue
            if attr.name not in labels_dict:
                continue
            labels[labels_dict[attr.name]] = from_attribute(attr)
    if isinstance(video_or_frame_labels, etav.VideoLabels):
        events_map = defaultdict(etae.VideoEventContainer)
        for event in video_or_frame_labels.events:
            if event.name not in labels_dict:
                continue
            events_map[labels_dict[event.name]].add(event)
        for (name, events) in events_map.items():
            labels[name] = from_video_events(events)
        return labels
    objects_map = defaultdict(etao.DetectedObjectContainer)
    for dobj in video_or_frame_labels.objects:
        if dobj.name not in labels_dict:
            continue
        objects_map[labels_dict[dobj.name]].add(dobj)
    for (name, objects) in objects_map.items():
        labels[name] = from_detected_objects(objects)
    polylines_map = defaultdict(etap.PolylineContainer)
    for polyline in video_or_frame_labels.polylines:
        if polyline.name not in labels_dict:
            continue
        polylines_map[labels_dict[polyline.name]].add(polyline)
    for (name, polylines) in polylines_map.items():
        labels[name] = from_polylines(polylines)
    keypoints_map = defaultdict(etak.KeypointsContainer)
    for keypoints in video_or_frame_labels.keypoints:
        if keypoints.name not in labels_dict:
            continue
        keypoints_map[labels_dict[keypoints.name]].add(keypoints)
    for (name, keypoints) in keypoints_map.items():
        labels[name] = from_keypoints(keypoints)
    if video_or_frame_labels.has_mask and 'mask' in labels_dict:
        labels['mask'] = fol.Segmentation(mask=video_or_frame_labels.mask)
    return labels

def _squeeze_extra_unit_dims(embeddings):
    if False:
        return 10
    dims = embeddings.shape[1:]
    extra_axes = tuple((ax for (ax, dim) in enumerate(dims, 1) if dim == 1))
    if len(extra_axes) == len(dims):
        extra_axes = extra_axes[1:]
    if extra_axes:
        return np.squeeze(embeddings, axis=extra_axes)
    return embeddings

def _add_logits(label, logits):
    if False:
        i = 10
        return i + 15
    if isinstance(label, fol.Classification):
        label.logits = logits[0]
    elif isinstance(label, fol.Classifications):
        for (c, l) in zip(label.classifications, logits):
            c.logits = l
    elif label is not None:
        msg = "Cannot store logits on label type '%s'" % label.__class__
        warnings.warn(msg)