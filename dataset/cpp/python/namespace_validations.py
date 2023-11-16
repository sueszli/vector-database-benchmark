"""API endpoint to do system level validations."""

from typing import Optional, Tuple

from flask import request
from flask_restx import Namespace, Resource

from app import errors as self_errors
from app import models, schema
from app.core import environments

api = Namespace("validations", description="Validates system requirements")
api = schema.register_schema(api)


def validate_environment(project_uuid: str, env_uuid: str) -> Tuple[str, Optional[str]]:
    """Validates whether the environments exist on the system.

    Only passes if the condition below is satisfied:
        * The image: ``_config.ENVIRONMENT_IMAGE_NAME`` exists in the
        registry.

    Args:
        project_uuid: Project UUID for which the environment should
            exist.
        env_uuid: Environment UUID to check.

    Returns:
        (check, action)

        `check` is "pass" or "fail".

        `action` is one of ["BUILD", "WAIT", "RETRY", None]

    """
    try:
        environments.get_env_uuids_to_image_mappings(project_uuid, {env_uuid})
    except self_errors.ImageNotFound:
        # Check the build history for the environment to determine the
        # action.
        env_builds = models.EnvironmentImageBuild.query.filter_by(
            project_uuid=project_uuid, environment_uuid=env_uuid
        )
        num_building_builds = env_builds.filter(
            models.EnvironmentImageBuild.status.in_(["PENDING", "STARTED"])
        ).count()

        if num_building_builds > 0:
            return "fail", "WAIT"

        num_failed_builds = env_builds.filter(
            models.EnvironmentImageBuild.status.in_(["FAILURE"])
        ).count()
        if num_failed_builds > 0:
            return "fail", "RETRY"

        return "fail", "BUILD"

    return "pass", None


@api.route("/environments")
class Gate(Resource):
    @api.doc("validate_environments")
    @api.expect(schema.validation_environments)
    @api.marshal_with(
        schema.validation_environments_result,
        code=201,
        description="Validation of environments",
    )
    def post(self):
        """Checks readiness of the given environments.

        Have the environments been built and are they ready.

        NOTE: The order of ``["fail"]`` and ``["action"]`` indicates the
        required action to convert the "fail" to a "pass".

        """
        post_data = request.get_json()
        environment_uuids = post_data["environment_uuids"]
        project_uuid = post_data["project_uuid"]

        res = {
            "validation": None,  # Will be set last
            "fail": [],
            "actions": [],
            "pass": [],
        }
        for env_uuid in environment_uuids:
            # Check will be either "fail" or "pass".
            validation, action = validate_environment(project_uuid, env_uuid)
            res[validation].append(env_uuid)

            if validation == "fail":
                res["actions"].append(action)

        res["validation"] = "fail" if len(res["fail"]) != 0 else "pass"
        return res, 201
