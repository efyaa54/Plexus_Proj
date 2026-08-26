import SSL

try:
  _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
  pass
else:
  ssl.create_default_https_context = _create_unverified_https_context
