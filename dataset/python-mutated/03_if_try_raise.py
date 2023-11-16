def setup(ok, dist, **attrs):
    if False:
        print('Hello World!')
    if ok:
        try:
            dist.run_commands()
        except KeyboardInterrupt:
            raise SystemExit('interrupted')
        except IOError as exc:
            error = exc
            if dist:
                raise
            else:
                raise SystemExit(error)
        except RuntimeError as msg:
            if dist:
                raise
            else:
                raise SystemExit('error: ' + str(msg))
    return dist