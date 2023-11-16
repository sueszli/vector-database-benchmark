from dataclasses import dataclass
from typing import Optional
LATEST_RELEASE_TTAS = {'android': 43747, 'apple': 1423869, 'apple-ios': 106006, 'apple-macos': 15176, 'capacitor': 14507, 'cocoa': 1141204, 'cocoa-objc': 1106061, 'cocoa-swift': 1217941, 'cordova': 18692, 'csharp': 16964, 'csharp-aspnetcore': 1730, 'dart': 14803, 'dotnet': 4350, 'dotnet-aspnet': 3734, 'dotnet-aspnetcore': 15197, 'dotnet-awslambda': 77201, 'dotnet-gcpfunctions': 2964, 'dotnet-maui': 5189, 'dotnet-uwp': 17688, 'dotnet-winforms': 3411, 'dotnet-wpf': 3610, 'dotnet-xamarin': 36899, 'electron': 25198, 'elixir': 12596, 'flutter': 204973, 'go': 16487, 'go-http': 5658, 'ionic': 7464, 'java': 12291, 'java-android': 608689, 'java-log4j2': 4733, 'java-logback': 51312, 'java-logging': 11843950, 'javascript': 11035, 'javascript-angular': 5156, 'javascript-angularjs': 11592, 'javascript-astro': 5943, 'javascript-backbone': 3463, 'javascript-browser': 3552, 'javascript-ember': 10246, 'javascript-gatsby': 5892, 'javascript-nextjs': 5840, 'javascript-performance-onboarding-2-configure': 5048, 'javascript-react': 5045, 'javascript-remix': 11751, 'javascript-replay-onboarding-1-install': 2609, 'javascript-svelte': 4604, 'javascript-sveltekit': 5222, 'javascript-vue': 10115, 'java-spring-boot': 11731, 'kotlin': 4520, 'minidump': 57271, 'native': 1687255, 'native-qt': 60835, 'node': 4072, 'node-awslambda': 14136, 'node-azurefunctions': 12385, 'node-connect': 4389, 'node-express': 4340, 'node-gcpfunctions': 3728, 'node-koa': 4214, 'node-nodeawslambda': 3639, 'node-serverlesscloud': 11155, 'objc': 4000209, 'other': 12019, 'php': 12112, 'php-laravel': 12382, 'php-symfony': 13781, 'python': 4358, 'python-aiohttp': 4048, 'python-asgi': 3999, 'python-awslambda': 11905, 'python-bottle': 7322, 'python-celery': 11376, 'python-chalice': 3933, 'python-django': 4482, 'python-falcon': 4206, 'python-fastapi': 4309, 'python-flask': 4507, 'python-gcpfunctions': 4597, 'python-pylons': 2982, 'python-pyramid': 12130, 'python-pythonawslambda': 3078, 'python-pythonserverless': 368842, 'python-quart': 4295, 'python-rq': 11121, 'python-sanic': 13805, 'python-serverless': 10475, 'python-starlette': 3689, 'python-tornado': 3483, 'python-tryton': 3605, 'python-wsgi': 4657, 'react-native': 54047, 'ruby': 3990, 'ruby-rack': 4599, 'ruby-rails': 4421, 'rust': 21803, 'swift': 1705490, 'unity': 4211, 'unreal': 15981}
DEFAULT_TTA = 3600

@dataclass(frozen=True)
class Platform:
    name: Optional[str] = None

    @property
    def time_to_adoption(self) -> int:
        if False:
            i = 10
            return i + 15
        return LATEST_RELEASE_TTAS.get(self.name, DEFAULT_TTA) if self.name else DEFAULT_TTA