import codecs


def as_bytes(bytes_or_text, encoding='utf-8'):
    """Converts `bytearray`, `bytes`, or unicode python input types to `bytes`.

  Uses utf-8 encoding for text by default.

  Args:
    bytes_or_text: A `bytearray`, `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for encoding unicode.

  Returns:
    A `bytes` object.

  Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
  """
    # Validate encoding, a LookupError will be raised if invalid.
    encoding = codecs.lookup(encoding).name
    if isinstance(bytes_or_text, bytearray):
        return bytes(bytes_or_text)
    elif isinstance(bytes_or_text, _six.text_type):
        return bytes_or_text.encode(encoding)
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text
    else:
        raise TypeError('Expected binary or unicode string, got %r' %
                        (bytes_or_text,))
