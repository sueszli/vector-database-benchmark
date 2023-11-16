from itertools import chain
from django import template
from django.contrib.contenttypes.models import ContentType
from django.template.defaultfilters import stringfilter
from django.utils import timezone
from django.utils.html import escape, conditional_escape
from django.utils.safestring import mark_safe, SafeData
from django.utils.text import normalize_newlines
from django.utils.translation import gettext as _
from django.urls import reverse
from django.contrib.auth.models import User
from dojo.utils import prepare_for_view, get_system_setting, get_full_url, get_file_images
import dojo.utils
from dojo.models import Check_List, FileAccessToken, Finding, System_Settings, Product, Dojo_User, Benchmark_Product
import markdown
from django.db.models import Sum, Case, When, IntegerField, Value
import dateutil.relativedelta
import datetime
import bleach
import git
from django.conf import settings
import dojo.jira_link.helper as jira_helper
import logging
logger = logging.getLogger(__name__)
register = template.Library()
markdown_tags = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'b', 'i', 'strong', 'em', 'tt', 'table', 'thead', 'th', 'tbody', 'tr', 'td', 'p', 'br', 'pre', 'div', 'span', 'blockquote', 'code', 'hr', 'ul', 'ol', 'li', 'dd', 'dt', 'img', 'a', 'sub', 'sup', 'center'}
markdown_attrs = {'*': ['id'], 'img': ['src', 'alt', 'title', 'width', 'height', 'style'], 'a': ['href', 'alt', 'target', 'title'], 'span': ['class'], 'pre': ['class'], 'div': ['class']}
markdown_styles = ['background-color']
finding_related_action_classes_dict = {'reset_finding_duplicate_status': 'fa-solid fa-eraser', 'set_finding_as_original': 'fa-brands fa-superpowers', 'mark_finding_duplicate': 'fa-solid fa-copy'}
finding_related_action_title_dict = {'reset_finding_duplicate_status': 'Reset duplicate status', 'set_finding_as_original': 'Set as original', 'mark_finding_duplicate': 'Mark as duplicate'}
supported_file_formats = ['apng', 'avif', 'gif', 'jpg', 'jpeg', 'jfif', 'pjpeg', 'pjp', 'png', 'svg', 'webp', 'pdf']

@register.filter
def markdown_render(value):
    if False:
        for i in range(10):
            print('nop')
    if value:
        markdown_text = markdown.markdown(value, extensions=['markdown.extensions.nl2br', 'markdown.extensions.sane_lists', 'markdown.extensions.codehilite', 'markdown.extensions.fenced_code', 'markdown.extensions.toc', 'markdown.extensions.tables'])
        return mark_safe(bleach.clean(markdown_text, tags=markdown_tags, attributes=markdown_attrs, css_sanitizer=markdown_styles))

@register.filter(name='url_shortner')
def url_shortner(value):
    if False:
        for i in range(10):
            print('nop')
    return_value = str(value)
    if len(return_value) > 50:
        return_value = '...' + return_value[-47:]
    return return_value

@register.filter(name='get_pwd')
def get_pwd(value):
    if False:
        i = 10
        return i + 15
    return prepare_for_view(value)

@register.filter(name='checklist_status')
def checklist_status(value):
    if False:
        i = 10
        return i + 15
    return Check_List.get_status(value)

@register.filter(is_safe=True, needs_autoescape=True)
@stringfilter
def linebreaksasciidocbr(value, autoescape=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts all newlines in a piece of plain text to HTML line breaks\n    (``+ <br />``).\n    '
    autoescape = autoescape and (not isinstance(value, SafeData))
    value = normalize_newlines(value)
    if autoescape:
        value = escape(value)
    return mark_safe(value.replace('\n', '&nbsp;+<br />'))

@register.simple_tag
def dojo_version():
    if False:
        print('Hello World!')
    from dojo import __version__
    version = __version__
    if settings.FOOTER_VERSION:
        version = settings.FOOTER_VERSION
    return 'v. {}'.format(version)

@register.simple_tag
def dojo_current_hash():
    if False:
        return 10
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha[:8]
    except:
        return 'release mode'

@register.simple_tag
def display_date():
    if False:
        while True:
            i = 10
    return timezone.localtime(timezone.now()).strftime('%b %d, %Y')

@register.simple_tag
def dojo_docs_url():
    if False:
        i = 10
        return i + 15
    from dojo import __docs__
    return mark_safe(__docs__)

@register.filter
def content_type(obj):
    if False:
        while True:
            i = 10
    if not obj:
        return False
    return ContentType.objects.get_for_model(obj).id

@register.filter
def content_type_str(obj):
    if False:
        i = 10
        return i + 15
    if not obj:
        return False
    return ContentType.objects.get_for_model(obj)

@register.filter(name='remove_string')
def remove_string(string, value):
    if False:
        i = 10
        return i + 15
    return string.replace(value, '')

@register.filter
def percentage(fraction, value):
    if False:
        print('Hello World!')
    return_value = ''
    if int(value) > 0:
        try:
            return_value = '%.1f%%' % (float(fraction) / float(value) * 100)
        except ValueError:
            pass
    return return_value

def asvs_calc_level(benchmark_score):
    if False:
        for i in range(10):
            print('nop')
    total = 0
    total_pass = 0
    total_fail = 0
    total_wait = 0
    total_viewed = 0
    if benchmark_score:
        benchmarks = Benchmark_Product.objects.filter(product_id=benchmark_score.product_id, enabled=True, control__category__type=benchmark_score.benchmark_type)
        if benchmark_score.desired_level == 'Level 1':
            benchmarks = benchmarks.filter(control__level_1=True)
        elif benchmark_score.desired_level == 'Level 2':
            benchmarks = benchmarks.filter(control__level_2=True)
        elif benchmark_score.desired_level == 'Level 3':
            benchmarks = benchmarks.filter(control__level_3=True)
        noted_benchmarks = benchmarks.filter(notes__isnull=False).order_by('id').distinct()
        noted_benchmarks_ids = [b.id for b in noted_benchmarks]
        total = len(benchmarks)
        total_pass = len([bench for bench in benchmarks if bench.pass_fail])
        total_fail = len([bench for bench in benchmarks if not bench.pass_fail and bench.id in noted_benchmarks_ids])
        total_wait = len([bench for bench in benchmarks if not bench.pass_fail and bench.id not in noted_benchmarks_ids])
        total_viewed = total_pass + total_fail
    return (benchmark_score.desired_level, total, total_pass, total_wait, total_fail, total_viewed)

@register.filter
def asvs_level(benchmark_score):
    if False:
        return 10
    (benchmark_score.desired_level, total, total_pass, total_wait, total_fail, total_viewed) = asvs_calc_level(benchmark_score)
    level = percentage(total_viewed, total)
    return _('Checklist is %(level)s full (pass: %(total_viewed)s, total: %(total)s)') % {'level': level, 'total_viewed': total_viewed, 'total': total}

@register.filter(name='version_num')
def version_num(value):
    if False:
        return 10
    version = ''
    if value:
        version = 'v.' + value
    return version

@register.filter(name='group_sla')
def group_sla(group):
    if False:
        i = 10
        return i + 15
    if not get_system_setting('enable_finding_sla'):
        return ''
    if not group.findings.all():
        return ''
    finding = group.findings.all().order_by('severity').first()
    return finding_sla(finding)

@register.filter(name='finding_sla')
def finding_sla(finding):
    if False:
        while True:
            i = 10
    if not get_system_setting('enable_finding_sla'):
        return ''
    title = ''
    severity = finding.severity
    find_sla = finding.sla_days_remaining()
    sla_age = getattr(finding.get_sla_periods(), severity.lower(), None)
    if finding.mitigated:
        status = 'blue'
        status_text = 'Remediated within SLA for ' + severity.lower() + ' findings (' + str(sla_age) + ' days since ' + finding.get_sla_start_date().strftime('%b %d, %Y') + ')'
        if find_sla and find_sla < 0:
            status = 'orange'
            find_sla = abs(find_sla)
            status_text = 'Out of SLA: Remediated ' + str(find_sla) + ' days past SLA for ' + severity.lower() + ' findings (' + str(sla_age) + ' days since ' + finding.get_sla_start_date().strftime('%b %d, %Y') + ')'
    else:
        status = 'green'
        status_text = 'Remediation for ' + severity.lower() + ' findings in ' + str(sla_age) + ' days or less since ' + finding.get_sla_start_date().strftime('%b %d, %Y')
        if find_sla and find_sla < 0:
            status = 'red'
            status_text = 'Overdue: Remediation for ' + severity.lower() + ' findings in ' + str(sla_age) + ' days or less since ' + finding.get_sla_start_date().strftime('%b %d, %Y')
    if find_sla is not None:
        title = '<a class="has-popover" data-toggle="tooltip" data-placement="bottom" title="" href="#" data-content="' + status_text + '"><span class="label severity age-' + status + '">' + str(find_sla) + '</span></a>'
    return mark_safe(title)

@register.filter(name='product_grade')
def product_grade(product):
    if False:
        i = 10
        return i + 15
    grade = ''
    system_settings = System_Settings.objects.get()
    if system_settings.enable_product_grade and product:
        prod_numeric_grade = product.prod_numeric_grade
        if prod_numeric_grade == '' or prod_numeric_grade is None:
            from dojo.utils import calculate_grade
            calculate_grade(product)
        if prod_numeric_grade:
            if prod_numeric_grade >= system_settings.product_grade_a:
                grade = 'A'
            elif prod_numeric_grade < system_settings.product_grade_a and prod_numeric_grade >= system_settings.product_grade_b:
                grade = 'B'
            elif prod_numeric_grade < system_settings.product_grade_b and prod_numeric_grade >= system_settings.product_grade_c:
                grade = 'C'
            elif prod_numeric_grade < system_settings.product_grade_c and prod_numeric_grade >= system_settings.product_grade_d:
                grade = 'D'
            elif prod_numeric_grade <= system_settings.product_grade_f:
                grade = 'F'
    return grade

@register.filter
def display_index(data, index):
    if False:
        while True:
            i = 10
    return data[index]

@register.filter(is_safe=True, needs_autoescape=False)
@stringfilter
def action_log_entry(value, autoescape=None):
    if False:
        while True:
            i = 10
    import json
    history = json.loads(value)
    text = ''
    for k in history.keys():
        text += k.capitalize() + ' changed from "' + history[k][0] + '" to "' + history[k][1] + '"\n'
    return text

@register.simple_tag(takes_context=True)
def dojo_body_class(context):
    if False:
        i = 10
        return i + 15
    request = context['request']
    return request.COOKIES.get('dojo-sidebar', 'min')

@register.filter(name='datediff_time')
def datediff_time(date1, date2):
    if False:
        for i in range(10):
            print('nop')
    date_str = ''
    diff = dateutil.relativedelta.relativedelta(date2, date1)
    attrs = ['years', 'months', 'days']
    human_readable = lambda delta: ['%d %s' % (getattr(delta, attr), getattr(delta, attr) > 1 and attr or attr[:-1]) for attr in attrs if getattr(delta, attr)]
    human_date = human_readable(diff)
    for date_part in human_date:
        date_str = date_str + date_part + ' '
    if date_str == '':
        date_str = '1 day'
    return date_str

@register.filter(name='overdue')
def overdue(date1):
    if False:
        for i in range(10):
            print('nop')
    date_str = ''
    if date1 < datetime.datetime.now().date():
        date_str = datediff_time(date1, datetime.datetime.now().date())
    return date_str

@register.filter(name='notspecified')
def notspecified(text):
    if False:
        while True:
            i = 10
    if text:
        return text
    else:
        return mark_safe('<em class="text-muted">Not Specified</em>')

@register.tag
def colgroup(parser, token):
    if False:
        while True:
            i = 10
    '\n    Usage:: {% colgroup items into 3 cols as grouped_items %}\n\n    <table border="0">\n        {% for row in grouped_items %}\n        <tr>\n            {% for item in row %}\n            <td>{% if item %}{{ forloop.parentloop.counter }}. {{ item }}{% endif %}</td>\n            {% endfor %}\n        </tr>\n        {% endfor %}\n    </table>\n\n    Outputs::\n    ============================================\n    | 1. One   | 1. Eleven   | 1. Twenty One   |\n    | 2. Two   | 2. Twelve   | 2. Twenty Two   |\n    | 3. Three | 3. Thirteen | 3. Twenty Three |\n    | 4. Four  | 4. Fourteen |                 |\n    ============================================\n    '

    class Node(template.Node):

        def __init__(self, iterable, num_cols, varname):
            if False:
                return 10
            self.iterable = iterable
            self.num_cols = num_cols
            self.varname = varname

        def render(self, context):
            if False:
                print('Hello World!')
            iterable = template.Variable(self.iterable).resolve(context)
            num_cols = self.num_cols
            context[self.varname] = zip(*[chain(iterable, [None] * (num_cols - 1))] * num_cols)
            return ''
    try:
        (_, iterable, _, num_cols, _, _, varname) = token.split_contents()
        num_cols = int(num_cols)
    except ValueError:
        raise template.TemplateSyntaxError('Invalid arguments passed to %r.' % token.contents.split()[0])
    return Node(iterable, num_cols, varname)

@register.simple_tag(takes_context=True)
def pic_token(context, image, size):
    if False:
        return 10
    user_id = context['user_id']
    user = User.objects.get(id=user_id)
    token = FileAccessToken(user=user, file=image, size=size)
    token.save()
    return reverse('download_finding_pic', args=[token.token])

@register.filter
def file_images(obj):
    if False:
        print('Hello World!')
    return get_file_images(obj, return_objects=True)

@register.simple_tag
def severity_number_value(value):
    if False:
        while True:
            i = 10
    return Finding.get_number_severity(value)

@register.filter
def tracked_object_value(current_object):
    if False:
        for i in range(10):
            print('nop')
    value = ''
    if current_object.path is not None:
        value = current_object.path
    elif current_object.folder is not None:
        value = current_object.folder
    elif current_object.artifact is not None:
        value = current_object.artifact
    return value

@register.filter
def tracked_object_type(current_object):
    if False:
        print('Hello World!')
    value = ''
    if current_object.path is not None:
        value = 'File'
    elif current_object.folder is not None:
        value = 'Folder'
    elif current_object.artifact is not None:
        value = 'Artifact'
    return value

def icon(name, tooltip):
    if False:
        print('Hello World!')
    return '<i class="fa-solid fa-' + name + ' has-popover" data-trigger="hover" data-placement="bottom" data-content="' + tooltip + '"></i>'

def not_specified_icon(tooltip):
    if False:
        i = 10
        return i + 15
    return '<i class="fa-solid fa-question fa-fw text-danger has-popover" aria-hidden="true" data-trigger="hover" data-placement="bottom" data-content="' + tooltip + '"></i>'

def stars(filled, total, tooltip):
    if False:
        for i in range(10):
            print('nop')
    code = '<i class="has-popover" data-placement="bottom" data-content="' + tooltip + '">'
    for i in range(0, total):
        if i < filled:
            code += '<i class="fa-solid fa-star has-popover" aria-hidden="true"></span>'
        else:
            code += '<i class="fa-regular fa-star text-muted has-popover" aria-hidden="true"></span>'
    code += '</i>'
    return code

@register.filter
def business_criticality_icon(value):
    if False:
        print('Hello World!')
    if value == Product.VERY_HIGH_CRITICALITY:
        return mark_safe(stars(5, 5, 'Very High'))
    if value == Product.HIGH_CRITICALITY:
        return mark_safe(stars(4, 5, 'High'))
    if value == Product.MEDIUM_CRITICALITY:
        return mark_safe(stars(3, 5, 'Medium'))
    if value == Product.LOW_CRITICALITY:
        return mark_safe(stars(2, 5, 'Low'))
    if value == Product.VERY_LOW_CRITICALITY:
        return mark_safe(stars(1, 5, 'Very Low'))
    if value == Product.NONE_CRITICALITY:
        return mark_safe(stars(0, 5, 'None'))
    else:
        return ''

@register.filter
def last_value(value):
    if False:
        return 10
    if '/' in value:
        return value.rsplit('/')[-1:][0]
    else:
        return value

@register.filter
def platform_icon(value):
    if False:
        while True:
            i = 10
    if value == Product.WEB_PLATFORM:
        return mark_safe(icon('list-alt', 'Web'))
    elif value == Product.DESKTOP_PLATFORM:
        return mark_safe(icon('desktop', 'Desktop'))
    elif value == Product.MOBILE_PLATFORM:
        return mark_safe(icon('mobile', 'Mobile'))
    elif value == Product.WEB_SERVICE_PLATFORM:
        return mark_safe(icon('plug', 'Web Service'))
    elif value == Product.IOT:
        return mark_safe(icon('random', 'Internet of Things'))
    else:
        return ''

@register.filter
def lifecycle_icon(value):
    if False:
        while True:
            i = 10
    if value == Product.CONSTRUCTION:
        return mark_safe(icon('compass', 'Explore'))
    if value == Product.PRODUCTION:
        return mark_safe(icon('ship', 'Sustain'))
    if value == Product.RETIREMENT:
        return mark_safe(icon('moon-o', 'Retire'))
    else:
        return ''

@register.filter
def origin_icon(value):
    if False:
        while True:
            i = 10
    if value == Product.THIRD_PARTY_LIBRARY_ORIGIN:
        return mark_safe(icon('book', 'Third-Party Library'))
    if value == Product.PURCHASED_ORIGIN:
        return mark_safe(icon('money', 'Purchased'))
    if value == Product.CONTRACTOR_ORIGIN:
        return mark_safe(icon('suitcase', 'Contractor Developed'))
    if value == Product.INTERNALLY_DEVELOPED_ORIGIN:
        return mark_safe(icon('home', 'Internally Developed'))
    if value == Product.OPEN_SOURCE_ORIGIN:
        return mark_safe(icon('code', 'Open Source'))
    if value == Product.OUTSOURCED_ORIGIN:
        return mark_safe(icon('globe', 'Outsourced'))
    else:
        return ''

@register.filter
def external_audience_icon(value):
    if False:
        for i in range(10):
            print('nop')
    if value:
        return mark_safe(icon('users', 'External Audience'))
    else:
        return ''

@register.filter
def internet_accessible_icon(value):
    if False:
        print('Hello World!')
    if value:
        return mark_safe(icon('cloud', 'Internet Accessible'))
    else:
        return ''

@register.filter
def get_severity_count(id, table):
    if False:
        print('Hello World!')
    if table == 'test':
        counts = Finding.objects.filter(test=id).prefetch_related('test__engagement__product').aggregate(total=Sum(Case(When(severity__in=('Critical', 'High', 'Medium', 'Low'), then=Value(1)), output_field=IntegerField())), critical=Sum(Case(When(severity='Critical', then=Value(1)), output_field=IntegerField())), high=Sum(Case(When(severity='High', then=Value(1)), output_field=IntegerField())), medium=Sum(Case(When(severity='Medium', then=Value(1)), output_field=IntegerField())), low=Sum(Case(When(severity='Low', then=Value(1)), output_field=IntegerField())), info=Sum(Case(When(severity='Info', then=Value(1)), output_field=IntegerField())))
    elif table == 'engagement':
        counts = Finding.objects.filter(test__engagement=id, active=True, duplicate=False).prefetch_related('test__engagement__product').aggregate(total=Sum(Case(When(severity__in=('Critical', 'High', 'Medium', 'Low'), then=Value(1)), output_field=IntegerField())), critical=Sum(Case(When(severity='Critical', then=Value(1)), output_field=IntegerField())), high=Sum(Case(When(severity='High', then=Value(1)), output_field=IntegerField())), medium=Sum(Case(When(severity='Medium', then=Value(1)), output_field=IntegerField())), low=Sum(Case(When(severity='Low', then=Value(1)), output_field=IntegerField())), info=Sum(Case(When(severity='Info', then=Value(1)), output_field=IntegerField())))
    elif table == 'product':
        counts = Finding.objects.filter(test__engagement__product=id).prefetch_related('test__engagement__product').aggregate(total=Sum(Case(When(severity__in=('Critical', 'High', 'Medium', 'Low'), then=Value(1)), output_field=IntegerField())), critical=Sum(Case(When(severity='Critical', then=Value(1)), output_field=IntegerField())), high=Sum(Case(When(severity='High', then=Value(1)), output_field=IntegerField())), medium=Sum(Case(When(severity='Medium', then=Value(1)), output_field=IntegerField())), low=Sum(Case(When(severity='Low', then=Value(1)), output_field=IntegerField())), info=Sum(Case(When(severity='Info', then=Value(1)), output_field=IntegerField())))
    critical = 0
    high = 0
    medium = 0
    low = 0
    info = 0
    if counts['info']:
        info = counts['info']
    if counts['low']:
        low = counts['low']
    if counts['medium']:
        medium = counts['medium']
    if counts['high']:
        high = counts['high']
    if counts['critical']:
        critical = counts['critical']
    total = critical + high + medium + low + info
    display_counts = []
    display_counts.append('Critical: ' + str(critical))
    display_counts.append('High: ' + str(high))
    display_counts.append('Medium: ' + str(medium))
    display_counts.append('Low: ' + str(low))
    display_counts.append('Info: ' + str(info))
    if table == 'test':
        display_counts.append('Total: ' + str(total) + ' Findings')
    elif table == 'engagement':
        display_counts.append('Total: ' + str(total) + ' Active Findings')
    elif table == 'product':
        display_counts.append('Total: ' + str(total) + ' Active Findings')
    display_counts = ', '.join([str(item) for item in display_counts])
    return display_counts

@register.filter
def full_url(url):
    if False:
        while True:
            i = 10
    return get_full_url(url)

@register.filter
def setting_enabled(name):
    if False:
        print('Hello World!')
    return getattr(settings, name, False)

@register.filter
def system_setting_enabled(name):
    if False:
        while True:
            i = 10
    return getattr(dojo.utils, name)()

@register.filter
def finding_display_status(finding):
    if False:
        for i in range(10):
            print('nop')
    display_status = finding.status()
    if 'Risk Accepted' in display_status:
        ra = finding.risk_acceptance
        if ra:
            url = reverse('view_risk_acceptance', args=(finding.test.engagement.id, ra.id))
            info = ra.name_and_expiration_info
            link = '<a href="' + url + '" class="has-popover" data-trigger="hover" data-placement="right" data-content="' + escape(info) + '" data-container="body" data-original-title="Risk Acceptance">Risk Accepted</a>'
            display_status = display_status.replace('Risk Accepted', link)
    if finding.under_review:
        url = reverse('defect_finding_review', args=(finding.id,))
        link = '<a href="' + url + '">Under Review</a>'
        display_status = display_status.replace('Under Review', link)
    if finding.duplicate:
        url = '#'
        name = 'unknown'
        if finding.duplicate_finding:
            url = reverse('view_finding', args=(finding.duplicate_finding.id,))
            name = finding.duplicate_finding.title + ', ' + finding.duplicate_finding.created.strftime('%b %d, %Y, %H:%M:%S')
        link = '<a href="' + url + '" data-toggle="tooltip" data-placement="top" title="' + escape(name) + '">Duplicate</a>'
        display_status = display_status.replace('Duplicate', link)
    return display_status

@register.filter
def cwe_url(cwe):
    if False:
        i = 10
        return i + 15
    if not cwe:
        return ''
    return 'https://cwe.mitre.org/data/definitions/' + str(cwe) + '.html'

@register.filter
def has_vulnerability_url(vulnerability_id):
    if False:
        while True:
            i = 10
    if not vulnerability_id:
        return False
    for key in settings.VULNERABILITY_URLS:
        if vulnerability_id.upper().startswith(key):
            return True
    return False

@register.filter
def vulnerability_url(vulnerability_id):
    if False:
        while True:
            i = 10
    if not vulnerability_id:
        return False
    for key in settings.VULNERABILITY_URLS:
        if vulnerability_id.upper().startswith(key):
            return settings.VULNERABILITY_URLS[key] + str(vulnerability_id)
    return ''

@register.filter
def first_vulnerability_id(finding):
    if False:
        return 10
    vulnerability_ids = finding.vulnerability_ids
    if vulnerability_ids:
        return vulnerability_ids[0]
    else:
        return None

@register.filter
def additional_vulnerability_ids(finding):
    if False:
        i = 10
        return i + 15
    vulnerability_ids = finding.vulnerability_ids
    if vulnerability_ids and len(vulnerability_ids) > 1:
        references = list()
        for vulnerability_id in vulnerability_ids[1:]:
            references.append(vulnerability_id)
        return references
    else:
        return None

@register.filter
def jiraencode(value):
    if False:
        for i in range(10):
            print('nop')
    if not value:
        return value
    return value.replace('|', '').replace('@', '')

@register.filter
def jiraencode_component(value):
    if False:
        while True:
            i = 10
    if not value:
        return value
    return value.replace('|', '').replace(':', ' : ').replace('@', ' @ ').replace('?', ' ? ').replace('#', ' # ')

@register.filter
def jira_project(obj, use_inheritance=True):
    if False:
        return 10
    return jira_helper.get_jira_project(obj, use_inheritance)

@register.filter
def jira_issue_url(obj):
    if False:
        i = 10
        return i + 15
    return jira_helper.get_jira_url(obj)

@register.filter
def jira_project_url(obj):
    if False:
        return 10
    return jira_helper.get_jira_project_url(obj)

@register.filter
def jira_key(obj):
    if False:
        for i in range(10):
            print('nop')
    return jira_helper.get_jira_key(obj)

@register.filter
def jira_creation(obj):
    if False:
        for i in range(10):
            print('nop')
    return jira_helper.get_jira_creation(obj)

@register.filter
def jira_change(obj):
    if False:
        i = 10
        return i + 15
    return jira_helper.get_jira_change(obj)

@register.filter
def get_thumbnail(file):
    if False:
        return 10
    from pathlib import Path
    file_format = Path(file.file.url).suffix[1:]
    return file_format in supported_file_formats

@register.filter
def finding_extended_title(finding):
    if False:
        while True:
            i = 10
    if not finding:
        return ''
    result = finding.title
    vulnerability_ids = finding.vulnerability_ids
    if vulnerability_ids:
        result += ' (' + vulnerability_ids[0] + ')'
    if finding.cwe:
        result += ' (CWE-' + str(finding.cwe) + ')'
    return result

@register.filter
def finding_duplicate_cluster_size(finding):
    if False:
        print('Hello World!')
    return len(finding.duplicate_finding_set()) + (1 if finding.duplicate_finding else 0)

@register.filter
def finding_related_action_classes(related_action):
    if False:
        return 10
    return finding_related_action_classes_dict.get(related_action, '')

@register.filter
def finding_related_action_title(related_action):
    if False:
        return 10
    return finding_related_action_title_dict.get(related_action, '')

@register.filter
def product_findings(product, findings):
    if False:
        print('Hello World!')
    return findings.filter(test__engagement__product=product).order_by('numerical_severity')

@register.filter
def class_name(value):
    if False:
        while True:
            i = 10
    return value.__class__.__name__

@register.filter(needs_autoescape=True)
def jira_project_tag(product_or_engagement, autoescape=True):
    if False:
        i = 10
        return i + 15
    if autoescape:
        esc = conditional_escape
    else:
        esc = lambda x: x
    jira_project = jira_helper.get_jira_project(product_or_engagement)
    if not jira_project:
        return ''
    html = '\n    <i class="fa %s has-popover %s"\n        title="<i class=\'fa %s\'></i> <b>JIRA Project Configuration%s</b>" data-trigger="hover" data-container="body" data-html="true" data-placement="bottom"\n        data-content="<b>Jira:</b> %s<br/>\n        <b>Project Key:</b> %s<br/>\n        <b>Component:</b> %s<br/>\n        <b>Push All Issues:</b> %s<br/>\n        <b>Engagement Epic Mapping:</b> %s<br/>\n        <b>Push Notes:</b> %s">\n    </i>\n    '
    jira_project_no_inheritance = jira_helper.get_jira_project(product_or_engagement, use_inheritance=False)
    inherited = True if not jira_project_no_inheritance else False
    icon = 'fa-bug'
    color = ''
    inherited_text = ''
    if inherited:
        color = 'lightgrey'
        inherited_text = ' (inherited)'
    if not jira_project.jira_instance:
        color = 'red'
        icon = 'fa-exclamation-triangle'
    return mark_safe(html % (icon, color, icon, inherited_text, esc(jira_project.jira_instance), esc(jira_project.project_key), esc(jira_project.component), esc(jira_project.push_all_issues), esc(jira_project.enable_engagement_epic_mapping), esc(jira_project.push_notes)))

@register.filter
def full_name(user):
    if False:
        for i in range(10):
            print('nop')
    return Dojo_User.generate_full_name(user)

@register.filter(needs_autoescape=True)
def import_settings_tag(test_import, autoescape=True):
    if False:
        for i in range(10):
            print('nop')
    if not test_import or not test_import.import_settings:
        return ''
    if autoescape:
        esc = conditional_escape
    else:
        esc = lambda x: x
    html = '\n\n    <i class="fa %s has-popover %s"\n        title="<i class=\'fa %s\'></i> <b>Import Settings</b>" data-trigger="hover" data-container="body" data-html="true" data-placement="bottom"\n        data-content="\n            <b>ID:</b> %s<br/>\n            <b>Active:</b> %s<br/>\n            <b>Verified:</b> %s<br/>\n            <b>Minimum Severity:</b> %s<br/>\n            <b>Close Old Findings:</b> %s<br/>\n            <b>Push to jira:</b> %s<br/>\n            <b>Tags:</b> %s<br/>\n            <b>Endpoints:</b> %s<br/>\n        "\n    </i>\n    '
    icon = 'fa-info-circle'
    color = ''
    return mark_safe(html % (icon, color, icon, esc(test_import.id), esc(test_import.import_settings.get('active', None)), esc(test_import.import_settings.get('verified', None)), esc(test_import.import_settings.get('minimum_severity', None)), esc(test_import.import_settings.get('close_old_findings', None)), esc(test_import.import_settings.get('push_to_jira', None)), esc(test_import.import_settings.get('tags', None)), esc(test_import.import_settings.get('endpoints', test_import.import_settings.get('endpoint', None)))))

@register.filter(needs_autoescape=True)
def import_history(finding, autoescape=True):
    if False:
        while True:
            i = 10
    if not finding or not settings.TRACK_IMPORT_HISTORY:
        return ''
    if autoescape:
        esc = conditional_escape
    else:
        esc = lambda x: x
    status_changes = finding.test_import_finding_action_set.all()
    if not status_changes or len(status_changes) < 2:
        return ''
    html = '\n\n    <i class="fa-solid fa-clock-rotate-left has-popover"\n        title="<i class=\'fa-solid fa-clock-rotate-left\'></i> <b>Import History</b>" data-trigger="hover" data-container="body" data-html="true" data-placement="right"\n        data-content="%s<br/>Currently only showing status changes made by import/reimport."\n    </i>\n    '
    list_of_status_changes = ''
    for status_change in status_changes:
        list_of_status_changes += '<b>' + status_change.created.strftime('%b %d, %Y, %H:%M:%S') + '</b>: ' + status_change.get_action_display() + '<br/>'
    return mark_safe(html % list_of_status_changes)