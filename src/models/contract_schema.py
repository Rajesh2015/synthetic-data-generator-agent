from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any


class QualityRule(BaseModel):
    type: str
    field: Optional[str] = None
    description: Optional[str] = None
    threshold: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    references: Optional[str] = None
    expression: Optional[str] = None


class FieldSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: str
    description: Optional[str] = None
    primaryKey: Optional[bool] = False
    unique: Optional[bool] = False
    required: Optional[bool] = False
    pattern: Optional[str] = None
    enum: Optional[List[str]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    # ODCS v3.1.0 exclusive range keywords
    exclusiveMinimum: Optional[float] = None
    exclusiveMaximum: Optional[float] = None
    minLength: Optional[int] = None
    maxLength: Optional[int] = None
    format: Optional[str] = None
    default: Optional[Any] = None
    references: Optional[str] = None
    x_fake: Optional[str] = Field(None, alias="x-fake")
    x_change_tracking: Optional[bool] = Field(None, alias="x-change-tracking")


class TableSchema(BaseModel):
    description: Optional[str] = None
    type: Optional[str] = "table"
    fields: Dict[str, FieldSchema]
    quality: Optional[List[QualityRule]] = None


class ContractInfo(BaseModel):
    title: str
    version: str
    description: Optional[str] = None
    owner: Optional[str] = None


class ParsedContract(BaseModel):
    id: str
    info: ContractInfo
    models: Dict[str, TableSchema]

    def get_primary_key(self, table_name: str) -> Optional[str]:
        table = self.models.get(table_name)
        if not table:
            return None
        for field_name, field in table.fields.items():
            if field.primaryKey:
                return field_name
        return None

    def get_foreign_keys(self, table_name: str) -> Dict[str, str]:
        table = self.models.get(table_name)
        if not table:
            return {}
        return {
            fname: field.references
            for fname, field in table.fields.items()
            if field.references
        }

    def get_change_tracked_fields(self, table_name: str) -> List[str]:
        table = self.models.get(table_name)
        if not table:
            return []
        return [
            fname for fname, field in table.fields.items()
            if field.x_change_tracking is True
        ]

    def get_dependency_order(self) -> List[str]:
        tables = list(self.models.keys())
        deps: Dict[str, set] = {t: set() for t in tables}

        for table_name in tables:
            for _, ref in self.get_foreign_keys(table_name).items():
                parent_table = ref.split(".")[0]
                if parent_table in deps:
                    deps[table_name].add(parent_table)

        ordered = []
        visited = set()

        def visit(table: str):
            if table in visited:
                return
            visited.add(table)
            for dep in deps.get(table, set()):
                visit(dep)
            ordered.append(table)

        for table in tables:
            visit(table)

        return ordered
