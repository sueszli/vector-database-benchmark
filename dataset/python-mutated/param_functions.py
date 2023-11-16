from typing import Any, Callable, List, Optional, Sequence
from prefect._vendor.fastapi import params
from prefect._internal.pydantic import HAS_PYDANTIC_V2
if HAS_PYDANTIC_V2:
    from pydantic.v1.fields import Undefined
else:
    from pydantic.fields import Undefined
from typing_extensions import Annotated, deprecated

def Path(default: Any=..., *, alias: Optional[str]=None, title: Optional[str]=None, description: Optional[str]=None, gt: Optional[float]=None, ge: Optional[float]=None, lt: Optional[float]=None, le: Optional[float]=None, min_length: Optional[int]=None, max_length: Optional[int]=None, regex: Optional[str]=None, examples: Optional[List[Any]]=None, example: Annotated[Optional[Any], deprecated('Deprecated in OpenAPI 3.1.0 that now uses JSON Schema 2020-12, although still supported. Use examples instead.')]=Undefined, deprecated: Optional[bool]=None, include_in_schema: bool=True, **extra: Any) -> Any:
    if False:
        while True:
            i = 10
    return params.Path(default=default, alias=alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, regex=regex, example=example, examples=examples, deprecated=deprecated, include_in_schema=include_in_schema, **extra)

def Query(default: Any=Undefined, *, alias: Optional[str]=None, title: Optional[str]=None, description: Optional[str]=None, gt: Optional[float]=None, ge: Optional[float]=None, lt: Optional[float]=None, le: Optional[float]=None, min_length: Optional[int]=None, max_length: Optional[int]=None, regex: Optional[str]=None, examples: Optional[List[Any]]=None, example: Annotated[Optional[Any], deprecated('Deprecated in OpenAPI 3.1.0 that now uses JSON Schema 2020-12, although still supported. Use examples instead.')]=Undefined, deprecated: Optional[bool]=None, include_in_schema: bool=True, **extra: Any) -> Any:
    if False:
        return 10
    return params.Query(default=default, alias=alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, regex=regex, example=example, examples=examples, deprecated=deprecated, include_in_schema=include_in_schema, **extra)

def Header(default: Any=Undefined, *, alias: Optional[str]=None, convert_underscores: bool=True, title: Optional[str]=None, description: Optional[str]=None, gt: Optional[float]=None, ge: Optional[float]=None, lt: Optional[float]=None, le: Optional[float]=None, min_length: Optional[int]=None, max_length: Optional[int]=None, regex: Optional[str]=None, examples: Optional[List[Any]]=None, example: Annotated[Optional[Any], deprecated('Deprecated in OpenAPI 3.1.0 that now uses JSON Schema 2020-12, although still supported. Use examples instead.')]=Undefined, deprecated: Optional[bool]=None, include_in_schema: bool=True, **extra: Any) -> Any:
    if False:
        print('Hello World!')
    return params.Header(default=default, alias=alias, convert_underscores=convert_underscores, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, regex=regex, example=example, examples=examples, deprecated=deprecated, include_in_schema=include_in_schema, **extra)

def Cookie(default: Any=Undefined, *, alias: Optional[str]=None, title: Optional[str]=None, description: Optional[str]=None, gt: Optional[float]=None, ge: Optional[float]=None, lt: Optional[float]=None, le: Optional[float]=None, min_length: Optional[int]=None, max_length: Optional[int]=None, regex: Optional[str]=None, examples: Optional[List[Any]]=None, example: Annotated[Optional[Any], deprecated('Deprecated in OpenAPI 3.1.0 that now uses JSON Schema 2020-12, although still supported. Use examples instead.')]=Undefined, deprecated: Optional[bool]=None, include_in_schema: bool=True, **extra: Any) -> Any:
    if False:
        i = 10
        return i + 15
    return params.Cookie(default=default, alias=alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, regex=regex, example=example, examples=examples, deprecated=deprecated, include_in_schema=include_in_schema, **extra)

def Body(default: Any=Undefined, *, embed: bool=False, media_type: str='application/json', alias: Optional[str]=None, title: Optional[str]=None, description: Optional[str]=None, gt: Optional[float]=None, ge: Optional[float]=None, lt: Optional[float]=None, le: Optional[float]=None, min_length: Optional[int]=None, max_length: Optional[int]=None, regex: Optional[str]=None, examples: Optional[List[Any]]=None, example: Annotated[Optional[Any], deprecated('Deprecated in OpenAPI 3.1.0 that now uses JSON Schema 2020-12, although still supported. Use examples instead.')]=Undefined, **extra: Any) -> Any:
    if False:
        i = 10
        return i + 15
    return params.Body(default=default, embed=embed, media_type=media_type, alias=alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, regex=regex, example=example, examples=examples, **extra)

def Form(default: Any=Undefined, *, media_type: str='application/x-www-form-urlencoded', alias: Optional[str]=None, title: Optional[str]=None, description: Optional[str]=None, gt: Optional[float]=None, ge: Optional[float]=None, lt: Optional[float]=None, le: Optional[float]=None, min_length: Optional[int]=None, max_length: Optional[int]=None, regex: Optional[str]=None, examples: Optional[List[Any]]=None, example: Annotated[Optional[Any], deprecated('Deprecated in OpenAPI 3.1.0 that now uses JSON Schema 2020-12, although still supported. Use examples instead.')]=Undefined, **extra: Any) -> Any:
    if False:
        i = 10
        return i + 15
    return params.Form(default=default, media_type=media_type, alias=alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, regex=regex, example=example, examples=examples, **extra)

def File(default: Any=Undefined, *, media_type: str='multipart/form-data', alias: Optional[str]=None, title: Optional[str]=None, description: Optional[str]=None, gt: Optional[float]=None, ge: Optional[float]=None, lt: Optional[float]=None, le: Optional[float]=None, min_length: Optional[int]=None, max_length: Optional[int]=None, regex: Optional[str]=None, examples: Optional[List[Any]]=None, example: Annotated[Optional[Any], deprecated('Deprecated in OpenAPI 3.1.0 that now uses JSON Schema 2020-12, although still supported. Use examples instead.')]=Undefined, **extra: Any) -> Any:
    if False:
        for i in range(10):
            print('nop')
    return params.File(default=default, media_type=media_type, alias=alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, regex=regex, example=example, examples=examples, **extra)

def Depends(dependency: Optional[Callable[..., Any]]=None, *, use_cache: bool=True) -> Any:
    if False:
        for i in range(10):
            print('nop')
    return params.Depends(dependency=dependency, use_cache=use_cache)

def Security(dependency: Optional[Callable[..., Any]]=None, *, scopes: Optional[Sequence[str]]=None, use_cache: bool=True) -> Any:
    if False:
        return 10
    return params.Security(dependency=dependency, scopes=scopes, use_cache=use_cache)