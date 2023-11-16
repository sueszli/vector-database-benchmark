# Copyright (c) 2015, Frappe Technologies Pvt. Ltd. and Contributors
# License: MIT. See LICENSE

import json
import os
import subprocess  # nosec

from semantic_version import Version

import frappe
from frappe import _, safe_decode
from frappe.utils import cstr


def get_change_log(user=None):
	if not user:
		user = frappe.session.user

	last_known_versions = frappe._dict(
		json.loads(frappe.db.get_value("User", user, "last_known_versions") or "{}")
	)
	current_versions = get_versions()

	if not last_known_versions:
		update_last_known_versions()
		return []

	change_log = []

	def set_in_change_log(app, opts, change_log):
		from_version = last_known_versions.get(app, {}).get("version") or "0.0.1"
		to_version = opts["version"]

		if from_version != to_version:
			app_change_log = get_change_log_for_app(app, from_version=from_version, to_version=to_version)

			if app_change_log:
				change_log.append(
					{
						"title": opts["title"],
						"description": opts["description"],
						"version": to_version,
						"change_log": app_change_log,
					}
				)

	for app, opts in current_versions.items():
		if app != "frappe":
			set_in_change_log(app, opts, change_log)

	if "frappe" in current_versions:
		set_in_change_log("frappe", current_versions["frappe"], change_log)

	return change_log


def get_change_log_for_app(app, from_version, to_version):
	change_log_folder = os.path.join(frappe.get_app_path(app), "change_log")
	if not os.path.exists(change_log_folder):
		return

	from_version = Version(from_version)
	to_version = Version(to_version)
	# remove pre-release part
	to_version.prerelease = None

	major_version_folders = [f"v{i}" for i in range(from_version.major, to_version.major + 1)]
	app_change_log = []

	for folder in os.listdir(change_log_folder):
		if folder in major_version_folders:
			for file in os.listdir(os.path.join(change_log_folder, folder)):
				version = Version(os.path.splitext(file)[0][1:].replace("_", "."))

				if from_version < version <= to_version:
					file_path = os.path.join(change_log_folder, folder, file)
					content = frappe.read_file(file_path)
					app_change_log.append([version, content])

	app_change_log = sorted(app_change_log, key=lambda d: d[0], reverse=True)

	# convert version to string and send
	return [[cstr(d[0]), d[1]] for d in app_change_log]


@frappe.whitelist()
def update_last_known_versions():
	frappe.db.set_value(
		"User",
		frappe.session.user,
		"last_known_versions",
		json.dumps(get_versions()),
		update_modified=False,
	)


@frappe.whitelist()
def get_versions():
	"""Get versions of all installed apps.

	Example:

	        {
	                "frappe": {
	                        "title": "Frappe Framework",
	                        "version": "5.0.0"
	                }
	        }"""
	versions = {}
	for app in frappe.get_installed_apps(_ensure_on_bench=True):
		app_hooks = frappe.get_hooks(app_name=app)
		versions[app] = {
			"title": app_hooks.get("app_title")[0],
			"description": app_hooks.get("app_description")[0],
			"branch": get_app_branch(app),
		}

		if versions[app]["branch"] != "master":
			branch_version = app_hooks.get("{}_version".format(versions[app]["branch"]))
			if branch_version:
				versions[app]["branch_version"] = branch_version[0] + " ({})".format(
					get_app_last_commit_ref(app)
				)

		try:
			versions[app]["version"] = frappe.get_attr(app + ".__version__")
		except AttributeError:
			versions[app]["version"] = "0.0.1"

	return versions


def get_app_branch(app):
	"""Returns branch of an app"""
	try:
		with open(os.devnull, "wb") as null_stream:
			result = subprocess.check_output(
				f"cd ../apps/{app} && git rev-parse --abbrev-ref HEAD",
				shell=True,
				stdin=null_stream,
				stderr=null_stream,
			)
		result = safe_decode(result)
		result = result.strip()
		return result
	except Exception:
		return ""


def get_app_last_commit_ref(app):
	try:
		with open(os.devnull, "wb") as null_stream:
			result = subprocess.check_output(
				f"cd ../apps/{app} && git rev-parse HEAD --short 7",
				shell=True,
				stdin=null_stream,
				stderr=null_stream,
			)
		result = safe_decode(result)
		result = result.strip()
		return result
	except Exception:
		return ""


def check_for_update():
	updates = frappe._dict(major=[], minor=[], patch=[])
	apps = get_versions()

	for app in apps:
		app_details = check_release_on_github(app)
		if not app_details:
			continue

		github_version, org_name = app_details
		# Get local instance's current version or the app

		branch_version = (
			apps[app]["branch_version"].split(" ", 1)[0] if apps[app].get("branch_version", "") else ""
		)
		instance_version = Version(branch_version or apps[app].get("version"))
		# Compare and popup update message
		for update_type in updates:
			if github_version.__dict__[update_type] > instance_version.__dict__[update_type]:
				updates[update_type].append(
					frappe._dict(
						current_version=str(instance_version),
						available_version=str(github_version),
						org_name=org_name,
						app_name=app,
						title=apps[app]["title"],
					)
				)
				break
			if github_version.__dict__[update_type] < instance_version.__dict__[update_type]:
				break

	add_message_to_redis(updates)


def parse_latest_non_beta_release(response):
	"""
	Parses the response JSON for all the releases and returns the latest non prerelease

	Parameters
	response (list): response object returned by github

	Returns
	json   : json object pertaining to the latest non-beta release
	"""
	version_list = [
		release.get("tag_name").strip("v") for release in response if not release.get("prerelease")
	]

	if version_list:
		return sorted(version_list, key=Version, reverse=True)[0]

	return None


def check_release_on_github(app: str):
	"""
	Check the latest release for a given Frappe application hosted on Github.

	Args:
	        app (str): The name of the Frappe application.

	Returns:
	        tuple(Version, str): The semantic version object of the latest release and the
	                organization name, if the application exists, otherwise None.
	"""

	import requests
	from giturlparse import parse
	from giturlparse.parser import ParserError

	try:
		# Check if repo remote is on github
		remote_url = subprocess.check_output(f"cd ../apps/{app} && git ls-remote --get-url", shell=True)
	except subprocess.CalledProcessError:
		# Passing this since some apps may not have git initialized in them
		return

	if isinstance(remote_url, bytes):
		remote_url = remote_url.decode()

	try:
		parsed_url = parse(remote_url)
	except ParserError:
		# Invalid URL
		return

	if parsed_url.resource != "github.com":
		return

	owner = parsed_url.owner
	repo = parsed_url.name

	# Get latest version from GitHub
	r = requests.get(f"https://api.github.com/repos/{owner}/{repo}/releases")
	if r.ok:
		latest_non_beta_release = parse_latest_non_beta_release(r.json())
		if latest_non_beta_release:
			return Version(latest_non_beta_release), owner


def add_message_to_redis(update_json):
	# "update-message" will store the update message string
	# "update-user-set" will be a set of users
	frappe.cache.set_value("update-info", json.dumps(update_json))
	user_list = [x.name for x in frappe.get_all("User", filters={"enabled": True})]
	system_managers = [user for user in user_list if "System Manager" in frappe.get_roles(user)]
	frappe.cache.sadd("update-user-set", *system_managers)


@frappe.whitelist()
def show_update_popup():
	user = frappe.session.user

	update_info = frappe.cache.get_value("update-info")
	if not update_info:
		return

	updates = json.loads(update_info)

	# Check if user is int the set of users to send update message to
	update_message = ""
	if frappe.cache.sismember("update-user-set", user):
		for update_type in updates:
			release_links = ""
			for app in updates[update_type]:
				app = frappe._dict(app)
				release_links += "<b>{title}</b>: <a href='https://github.com/{org_name}/{app_name}/releases/tag/v{available_version}'>v{available_version}</a><br>".format(
					available_version=app.available_version,
					org_name=app.org_name,
					app_name=app.app_name,
					title=app.title,
				)
			if release_links:
				message = _("New {} releases for the following apps are available").format(_(update_type))
				update_message += (
					"<div class='new-version-log'>{}<div class='new-version-links'>{}</div></div>".format(
						message, release_links
					)
				)

	if update_message:
		frappe.msgprint(update_message, title=_("New updates are available"), indicator="green")
		frappe.cache.srem("update-user-set", user)
