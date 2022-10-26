# @title Random Brightness Layer
import tensorflow as tf
from keras import backend
from keras.engine import base_layer
from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import preprocessing_utils as utils
from keras.utils import tf_utils

from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls


@keras_export("keras.__internal__.layers.BaseImageAugmentationLayer")
class BaseImageAugmentationLayer(base_layer.BaseRandomLayer):
    """Abstract base layer for image augmentaion.
    This layer contains base functionalities for preprocessing layers which
    augment image related data, eg. image and in future, label and bounding boxes.
    The subclasses could avoid making certain mistakes and reduce code
    duplications.
    This layer requires you to implement one method: `augment_image()`, which
    augments one single image during the training. There are a few additional
    methods that you can implement for added functionality on the layer:
    `augment_label()`, which handles label augmentation if the layer supports
    that.
    `augment_bounding_box()`, which handles the bounding box augmentation, if the
    layer supports that.
    `get_random_transformation()`, which should produce a random transformation
    setting. The tranformation object, which could be any type, will be passed to
    `augment_image`, `augment_label` and `augment_bounding_box`, to coodinate
    the randomness behavior, eg, in the RandomFlip layer, the image and
    bounding_box should be changed in the same way.
    The `call()` method support two formats of inputs:
    1. Single image tensor with 3D (HWC) or 4D (NHWC) format.
    2. A dict of tensors with stable keys. The supported keys are:
      `"images"`, `"labels"` and `"bounding_boxes"` at the moment. We might add
      more keys in future when we support more types of augmentation.
    The output of the `call()` will be in two formats, which will be the same
    structure as the inputs.
    The `call()` will handle the logic detecting the training/inference
    mode, unpack the inputs, forward to the correct function, and pack the output
    back to the same structure as the inputs.
    By default the `call()` method leverages the `tf.vectorized_map()` function.
    Auto-vectorization can be disabled by setting `self.auto_vectorize = False`
    in your `__init__()` method.  When disabled, `call()` instead relies
    on `tf.map_fn()`. For example:
    ```python
    class SubclassLayer(BaseImageAugmentationLayer):
      def __init__(self):
        super().__init__()
        self.auto_vectorize = False
    ```
    Example:
    ```python
    class RandomContrast(BaseImageAugmentationLayer):
      def __init__(self, factor=(0.5, 1.5), **kwargs):
        super().__init__(**kwargs)
        self._factor = factor
      def augment_image(self, image, transformation=None):
        random_factor = tf.random.uniform([], self._factor[0], self._factor[1])
        mean = tf.math.reduced_mean(inputs, axis=-1, keep_dim=True)
        return (inputs - mean) * random_factor + mean
    ```
    Note that since the randomness is also a common functionnality, this layer
    also includes a tf.keras.backend.RandomGenerator, which can be used to produce
    the random numbers.  The random number generator is stored in the
    `self._random_generator` attribute.
    """

    def __init__(self, rate=1.0, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.rate = rate

    @property
    def auto_vectorize(self):
        """Control whether automatic vectorization occurs.
        By default the `call()` method leverages the `tf.vectorized_map()` function.
        Auto-vectorization can be disabled by setting `self.auto_vectorize = False`
        in your `__init__()` method.  When disabled, `call()` instead relies
        on `tf.map_fn()`. For example:
        ```python
        class SubclassLayer(BaseImageAugmentationLayer):
          def __init__(self):
            super().__init__()
            self.auto_vectorize = False
        ```
        """
        return getattr(self, "_auto_vectorize", True)

    @auto_vectorize.setter
    def auto_vectorize(self, auto_vectorize):
        self._auto_vectorize = auto_vectorize

    @property
    def _map_fn(self):
        if self.auto_vectorize:
            return tf.vectorized_map
        else:
            return tf.map_fn

    @doc_controls.for_subclass_implementers
    def augment_image(self, image, transformation=None):
        """Augment a single image during training.
        Args:
          image: 3D image input tensor to the layer. Forwarded from `layer.call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness between
            image, label and bounding box.
        Returns:
          output 3D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    @doc_controls.for_subclass_implementers
    def augment_label(self, label, transformation=None):
        """Augment a single label during training.
        Args:
          label: 1D label to the layer. Forwarded from `layer.call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness between
            image, label and bounding box.
        Returns:
          output 1D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    @doc_controls.for_subclass_implementers
    def augment_bounding_box(self, bounding_box, transformation=None):
        """Augment bounding boxes for one image during training.
        Args:
          bounding_box: 2D bounding boxes to the layer. Forwarded from `call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness between
            image, label and bounding box.
        Returns:
          output 2D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    @doc_controls.for_subclass_implementers
    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        """Produce random transformation config for one single input.
        This is used to produce same randomness between image/label/bounding_box.
        Args:
          image: 3D image tensor from inputs.
          label: optional 1D label tensor from inputs.
          bounding_box: optional 2D bounding boxes tensor from inputs.
        Returns:
          Any type of object, which will be forwarded to `augment_image`,
          `augment_label` and `augment_bounding_box` as the `transformation`
          parameter.
        """
        return None

    def call(self, inputs, training=True):
        inputs = self._ensure_inputs_are_compute_dtype(inputs)
        if training:
            inputs, is_dict = self._format_inputs(inputs)
            images = inputs["images"]
            if images.shape.rank == 3:
                return self._format_output(self._augment(inputs), is_dict)
            elif images.shape.rank == 4:
                return self._format_output(self._batch_augment(inputs), is_dict)
            else:
                raise ValueError(
                    "Image augmentation layers are expecting inputs to be "
                    "rank 3 (HWC) or 4D (NHWC) tensors. Got shape: "
                    f"{images.shape}"
                )
        else:
            return inputs

    def _augment(self, inputs):
        image = inputs.get("images", None)
        label = inputs.get("labels", None)
        bounding_box = inputs.get("bounding_boxes", None)
        transformation = self.get_random_transformation(
            image=image, label=label, bounding_box=bounding_box
        )  # pylint: disable=assignment-from-none
        image = self.augment_image(image, transformation=transformation)
        result = {"images": image}
        if label is not None:
            label = self.augment_label(label, transformation=transformation)
            result["labels"] = label
        if bounding_box is not None:
            bounding_box = self.augment_bounding_box(
                bounding_box, transformation=transformation
            )
            result["bounding_boxes"] = bounding_box
        return result

    def _batch_augment(self, inputs):
        return self._map_fn(self._augment, inputs)

    def _format_inputs(self, inputs):
        if tf.is_tensor(inputs):
            # single image input tensor
            return {"images": inputs}, False
        elif isinstance(inputs, dict):
            # TODO(scottzhu): Check if it only contains the valid keys
            return inputs, True
        else:
            raise ValueError(
                f"Expect the inputs to be image tensor or dict. Got {inputs}"
            )

    def _format_output(self, output, is_dict):
        if not is_dict:
            return output["images"]
        else:
            return output

    def _ensure_inputs_are_compute_dtype(self, inputs):
        if isinstance(inputs, dict):
            inputs["images"] = utils.ensure_tensor(inputs["images"], self.compute_dtype)
        else:
            inputs = utils.ensure_tensor(inputs, self.compute_dtype)
        return inputs


@keras_export("keras.layers.RandomBrightness", v1=[])
class RandomBrightness(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly adjusts brightness during training.
    This layer will randomly increase/reduce the brightness for the input RGB
    images. At inference time, the output will be identical to the input.
    Call the layer with `training=True` to adjust the brightness of the input.
    Note that different brightness adjustment factors
    will be apply to each the images in the batch.
    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Args:
      factor: Float or a list/tuple of 2 floats between -1.0 and 1.0. The
        factor is used to determine the lower bound and upper bound of the
        brightness adjustment. A float value will be chosen randomly between
        the limits. When -1.0 is chosen, the output image will be black, and
        when 1.0 is chosen, the image will be fully white. When only one float
        is provided, eg, 0.2, then -0.2 will be used for lower bound and 0.2
        will be used for upper bound.
      value_range: Optional list/tuple of 2 floats for the lower and upper limit
        of the values of the input data. Defaults to [0.0, 255.0]. Can be changed
        to e.g. [0.0, 1.0] if the image input has been scaled before this layer.
        The brightness adjustment will be scaled to this range, and the
        output values will be clipped to this range.
      seed: optional integer, for fixed RNG behavior.
    Inputs: 3D (HWC) or 4D (NHWC) tensor, with float or int dtype. Input pixel
      values can be of any range (e.g. `[0., 1.)` or `[0, 255]`)
    Output: 3D (HWC) or 4D (NHWC) tensor with brightness adjusted based on the
      `factor`. By default, the layer will output floats. The output value will
      be clipped to the range `[0, 255]`, the valid range of RGB colors, and
      rescaled based on the `value_range` if needed.
    Sample usage:
    ```python
    random_bright = tf.keras.layers.RandomBrightness(factor=0.2)
    # An image with shape [2, 2, 3]
    image = [[[1, 2, 3], [4 ,5 ,6]], [[7, 8, 9], [10, 11, 12]]]
    # Assume we randomly select the factor to be 0.1, then it will apply
    # 0.1 * 255 to all the channel
    output = random_bright(image, training=True)
    # output will be int64 with 25.5 added to each channel and round down.
    tf.Tensor([[[26.5, 27.5, 28.5]
                [29.5, 30.5, 31.5]]
               [[32.5, 33.5, 34.5]
                [35.5, 36.5, 37.5]]],
              shape=(2, 2, 3), dtype=int64)
    ```
    """

    _FACTOR_VALIDATION_ERROR = (
        "The `factor` argument should be a number (or a list of two numbers) "
        "in the range [-1.0, 1.0]. "
    )
    _VALUE_RANGE_VALIDATION_ERROR = (
        "The `value_range` argument should be a list of two numbers. "
    )

    def __init__(self, factor, value_range=(0, 255), seed=None, **kwargs):
        base_preprocessing_layer.keras_kpl_gauge.get_cell("RandomBrightness").set(True)
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self._set_factor(factor)
        self._set_value_range(value_range)
        self._seed = seed

    def augment_image(self, image, transformation=None):
        return self._brightness_adjust(image, transformation["rgb_delta"])

    def augment_label(self, label, transformation=None):
        return label

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        rgb_delta_shape = (1, 1, 1)
        random_rgb_delta = self._random_generator.random_uniform(
            shape=rgb_delta_shape,
            minval=self._factor[0],
            maxval=self._factor[1],
        )
        random_rgb_delta = random_rgb_delta * (
            self._value_range[1] - self._value_range[0]
        )
        return {"rgb_delta": random_rgb_delta}

    def _set_value_range(self, value_range):
        if not isinstance(value_range, (tuple, list)):
            raise ValueError(self._VALUE_RANGE_VALIDATION_ERROR + f"Got {value_range}")
        if len(value_range) != 2:
            raise ValueError(self._VALUE_RANGE_VALIDATION_ERROR + f"Got {value_range}")
        self._value_range = sorted(value_range)

    def _set_factor(self, factor):
        if isinstance(factor, (tuple, list)):
            if len(factor) != 2:
                raise ValueError(self._FACTOR_VALIDATION_ERROR + f"Got {factor}")
            self._check_factor_range(factor[0])
            self._check_factor_range(factor[1])
            self._factor = sorted(factor)
        elif isinstance(factor, (int, float)):
            self._check_factor_range(factor)
            factor = abs(factor)
            self._factor = [-factor, factor]
        else:
            raise ValueError(self._FACTOR_VALIDATION_ERROR + f"Got {factor}")

    def _check_factor_range(self, input_number):
        if input_number > 1.0 or input_number < -1.0:
            raise ValueError(self._FACTOR_VALIDATION_ERROR + f"Got {input_number}")

    def _brightness_adjust(self, image, rgb_delta):
        image = utils.ensure_tensor(image, self.compute_dtype)
        rank = image.shape.rank
        if rank != 3:
            raise ValueError(
                "Expected the input image to be rank 3. Got "
                f"inputs.shape = {image.shape}"
            )
        rgb_delta = tf.cast(rgb_delta, image.dtype)
        image += rgb_delta
        return tf.clip_by_value(image, self._value_range[0], self._value_range[1])

    def get_config(self):
        config = {
            "factor": self._factor,
            "value_range": self._value_range,
            "seed": self._seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
