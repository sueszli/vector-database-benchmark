urlpatterns = []

class HandlerView:

    @classmethod
    def as_view(cls):
        if False:
            i = 10
            return i + 15

        def view():
            if False:
                i = 10
                return i + 15
            pass
        return view
handler400 = HandlerView.as_view()
handler403 = HandlerView.as_view()
handler404 = HandlerView.as_view()
handler500 = HandlerView.as_view()