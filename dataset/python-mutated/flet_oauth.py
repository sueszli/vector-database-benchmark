from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from flet_fastapi.flet_app_manager import app_manager
from flet_fastapi.oauth_state import OAuthState

class FletOAuth:
    """
    HTTP handler for OAuth callback.
    """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    async def handle(self, request: Request):
        """
        Handle OAuth callback request.

        Request must contain minimum `code` and `state` query parameters.

        Returns either redirect to a Flet page or a HTML page with further instructions.
        """
        state_id = request.query_params.get('state')
        if not state_id:
            raise HTTPException(status_code=400, detail='Invalid state')
        state = await app_manager.retrieve_state(state_id)
        if not state:
            raise HTTPException(status_code=400, detail='Invalid state')
        session = await app_manager.get_session(state.session_id)
        if not session:
            raise HTTPException(status_code=500, detail='Session not found')
        await session._authorize_callback_async({'state': state_id, 'code': request.query_params.get('code'), 'error': request.query_params.get('error'), 'error_description': request.query_params.get('error_description')})
        if state.complete_page_url:
            return RedirectResponse(state.complete_page_url)
        else:
            html_content = state.complete_page_html if state.complete_page_html else f"""\n            <!DOCTYPE html>\n            <html>\n            <head>\n                <title>Signed in successfully</title>\n            </head>\n            <body>\n                <script type="text/javascript">\n                    window.close();\n                </script>\n                <p>You've been successfully signed in! You can close this tab or window now.</p>\n            </body>\n            </html>\n            """
            return HTMLResponse(content=html_content, status_code=200)