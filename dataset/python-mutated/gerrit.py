from buildbot.steps.source.git import Git

class Gerrit(Git):

    def run_vc(self, branch, revision, patch):
        if False:
            print('Hello World!')
        gerrit_branch = None
        changed_project = self.build.getProperty('event.change.project')
        if not self.sourcestamp or self.sourcestamp.project != changed_project:
            pass
        elif self.build.hasProperty('event.patchSet.ref'):
            gerrit_branch = self.build.getProperty('event.patchSet.ref')
            self.updateSourceProperty('gerrit_branch', gerrit_branch)
        else:
            try:
                change = self.build.getProperty('gerrit_change', '').split('/')
                if len(change) == 2:
                    gerrit_branch = f'refs/changes/{int(change[0]) % 100:2}/{int(change[0])}/{int(change[1])}'
                    self.updateSourceProperty('gerrit_branch', gerrit_branch)
            except Exception:
                pass
        branch = gerrit_branch or branch
        return super().run_vc(branch, revision, patch)