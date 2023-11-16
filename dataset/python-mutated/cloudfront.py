from . import network_port, tags_or_list

def validate_network_port(x):
    if False:
        print('Hello World!')
    '\n    Property: CustomOriginConfig.HTTPPort\n    Property: CustomOriginConfig.HTTPSPort\n    '
    return network_port(x)

def validate_tags_or_list(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: Distribution.Tags\n    Property: StreamingDistribution.Tags\n    '
    return tags_or_list(x)

def cloudfront_access_control_allow_methods(access_control_allow_methods):
    if False:
        print('Hello World!')
    '\n    Property: AccessControlAllowMethods.Items\n    '
    valid_values = ['GET', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH', 'POST', 'PUT', 'ALL']
    if not isinstance(access_control_allow_methods, list):
        raise TypeError('AccessControlAllowMethods is not a list')
    for method in access_control_allow_methods:
        if method not in valid_values:
            raise ValueError('AccessControlAllowMethods must be one of: "%s"' % ', '.join(valid_values))
    return access_control_allow_methods

def cloudfront_cache_cookie_behavior(cookie_behavior):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: CacheCookiesConfig.CookieBehavior\n    '
    valid_values = ['none', 'whitelist', 'allExcept', 'all']
    if cookie_behavior not in valid_values:
        raise ValueError('CookieBehavior must be one of: "%s"' % ', '.join(valid_values))
    return cookie_behavior

def cloudfront_cache_header_behavior(header_behavior):
    if False:
        print('Hello World!')
    '\n    Property: CacheHeadersConfig.HeaderBehavior\n    '
    valid_values = ['none', 'whitelist']
    if header_behavior not in valid_values:
        raise ValueError('HeaderBehavior must be one of: "%s"' % ', '.join(valid_values))
    return header_behavior

def cloudfront_cache_query_string_behavior(query_string_behavior):
    if False:
        i = 10
        return i + 15
    '\n    Property: CacheQueryStringsConfig.QueryStringBehavior\n    '
    valid_values = ['none', 'whitelist', 'allExcept', 'all']
    if query_string_behavior not in valid_values:
        raise ValueError('QueryStringBehavior must be one of: "%s"' % ', '.join(valid_values))
    return query_string_behavior

def cloudfront_event_type(event_type):
    if False:
        return 10
    '\n    Property: LambdaFunctionAssociation.EventType\n    '
    valid_values = ['viewer-request', 'viewer-response', 'origin-request', 'origin-response']
    if event_type not in valid_values:
        raise ValueError('EventType must be one of: "%s"' % ', '.join(valid_values))
    return event_type

def cloudfront_forward_type(forward):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: Cookies.Forward\n    '
    valid_values = ['none', 'all', 'whitelist']
    if forward not in valid_values:
        raise ValueError('Forward must be one of: "%s"' % ', '.join(valid_values))
    return forward

def cloudfront_frame_option(frame_option):
    if False:
        while True:
            i = 10
    '\n    Property: FrameOptions.FrameOption\n    '
    valid_values = ['DENY', 'SAMEORIGIN']
    if frame_option not in valid_values:
        raise ValueError('FrameOption must be of: "%s"' % ', '.join(valid_values))
    return frame_option

def cloudfront_origin_request_cookie_behavior(cookie_behavior):
    if False:
        print('Hello World!')
    '\n    Property: OriginRequestCookiesConfig.CookieBehavior\n    '
    valid_values = ['none', 'whitelist', 'all', 'allExcept']
    if cookie_behavior not in valid_values:
        raise ValueError('CookieBehavior must be one of: "%s"' % ', '.join(valid_values))
    return cookie_behavior

def cloudfront_origin_request_header_behavior(header_behavior):
    if False:
        return 10
    '\n    Property: OriginRequestHeadersConfig.HeaderBehavior\n    '
    valid_values = ['none', 'whitelist', 'allViewer', 'allViewerAndWhitelistCloudFront', 'allExcept']
    if header_behavior not in valid_values:
        raise ValueError('HeaderBehavior must be one of: "%s"' % ', '.join(valid_values))
    return header_behavior

def cloudfront_origin_request_query_string_behavior(query_string_behavior):
    if False:
        return 10
    '\n    Property: OriginRequestQueryStringsConfig.QueryStringBehavior\n    '
    valid_values = ['none', 'whitelist', 'all', 'allExcept']
    if query_string_behavior not in valid_values:
        raise ValueError('QueryStringBehavior must be one of: "%s"' % ', '.join(valid_values))
    return query_string_behavior

def cloudfront_referrer_policy(referrer_policy):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: ReferrerPolicy.ReferrerPolicy\n    '
    valid_values = ['no-referrer', 'no-referrer-when-downgrade', 'origin', 'origin-when-cross-origin', 'same-origin', 'strict-origin', 'strict-origin-when-cross-origin', 'unsafe-url']
    if referrer_policy not in valid_values:
        raise ValueError('ReferrerPolicy must be of: "%s"' % ', '.join(valid_values))
    return referrer_policy

def cloudfront_restriction_type(restriction_type):
    if False:
        print('Hello World!')
    '\n    Property: GeoRestriction.RestrictionType\n    '
    valid_values = ['none', 'blacklist', 'whitelist']
    if restriction_type not in valid_values:
        raise ValueError('RestrictionType must be one of: "%s"' % ', '.join(valid_values))
    return restriction_type

def cloudfront_viewer_protocol_policy(viewer_protocol_policy):
    if False:
        while True:
            i = 10
    '\n    Property: CacheBehavior.ViewerProtocolPolicy\n    Property: DefaultCacheBehavior.ViewerProtocolPolicy\n    '
    valid_values = ['allow-all', 'redirect-to-https', 'https-only']
    if viewer_protocol_policy not in valid_values:
        raise ValueError('ViewerProtocolPolicy must be one of: "%s"' % ', '.join(valid_values))
    return viewer_protocol_policy

def priceclass_type(price_class):
    if False:
        print('Hello World!')
    '\n    Property: DistributionConfig.PriceClass\n    Property: StreamingDistributionConfig.PriceClass\n    '
    valid_values = ['PriceClass_100', 'PriceClass_200', 'PriceClass_All']
    if price_class not in valid_values:
        raise ValueError('PriceClass must be one of: "%s"' % ', '.join(valid_values))
    return price_class