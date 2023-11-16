from events.app_event import app_model_config_was_updated
from extensions.ext_database import db
from models.dataset import AppDatasetJoin
from models.model import AppModelConfig


@app_model_config_was_updated.connect
def handle(sender, **kwargs):
    app_model = sender
    app_model_config = kwargs.get('app_model_config')

    dataset_ids = get_dataset_ids_from_model_config(app_model_config)

    app_dataset_joins = db.session.query(AppDatasetJoin).filter(
        AppDatasetJoin.app_id == app_model.id
    ).all()

    removed_dataset_ids = []
    if not app_dataset_joins:
        added_dataset_ids = dataset_ids
    else:
        old_dataset_ids = set()
        for app_dataset_join in app_dataset_joins:
            old_dataset_ids.add(app_dataset_join.dataset_id)

        added_dataset_ids = dataset_ids - old_dataset_ids
        removed_dataset_ids = old_dataset_ids - dataset_ids

    if removed_dataset_ids:
        for dataset_id in removed_dataset_ids:
            db.session.query(AppDatasetJoin).filter(
                AppDatasetJoin.app_id == app_model.id,
                AppDatasetJoin.dataset_id == dataset_id
            ).delete()

    if added_dataset_ids:
        for dataset_id in added_dataset_ids:
            app_dataset_join = AppDatasetJoin(
                app_id=app_model.id,
                dataset_id=dataset_id
            )
            db.session.add(app_dataset_join)

    db.session.commit()


def get_dataset_ids_from_model_config(app_model_config: AppModelConfig) -> set:
    dataset_ids = set()
    if not app_model_config:
        return dataset_ids

    agent_mode = app_model_config.agent_mode_dict
    if agent_mode.get('enabled') is False:
        return dataset_ids

    if not agent_mode.get('tools'):
        return dataset_ids

    tools = agent_mode.get('tools')
    for tool in tools:
        tool_type = list(tool.keys())[0]
        tool_config = list(tool.values())[0]
        if tool_type == "dataset":
            dataset_ids.add(tool_config.get("id"))

    return dataset_ids
