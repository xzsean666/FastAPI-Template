from sqlalchemy import Column, PrimaryKeyConstraint, String, create_engine, inspect
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

# 基类
Base = declarative_base()


# 将数据库结果转换为字典
def to_dict(db_result):
    data = vars(db_result)
    data.pop("_sa_instance_state", None)
    return data


# 键值数据库类
class KVDB:
    def __init__(self, postgres_connection, table_name, pool_size=5):
        self.db_url = postgres_connection
        self.db = create_engine(self.db_url, poolclass=QueuePool, pool_size=pool_size)
        self.Session = sessionmaker(bind=self.db)
        self.model_class = self.create_cache_model(table_name)
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        inspector = inspect(self.db)
        if self.model_class.__tablename__ not in inspector.get_table_names():
            Base.metadata.create_all(self.db)

    def get_session(self):
        return self.Session()

    def get(self, key: str):
        with self.get_session() as session:
            cache = session.query(self.model_class).get(key)
            return cache.value if cache else None

    def put(self, key: str, value: dict):
        with self.get_session() as session:
            try:
                session.merge(self.model_class(key=key, value=value))
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                print(f"Error: {e}")
                return False

    def delete(self, key: str):
        with self.get_session() as session:
            try:
                cache = session.query(self.model_class).filter(
                    self.model_class.key == key
                )
                cache.delete()
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                print(f"Error: {e}")
                return False

    def keys(self):
        with self.get_session() as session:
            try:
                keys = session.query(self.model_class.key).all()
                return [key[0] for key in keys]
            except Exception as e:
                print(f"Error: {e}")
                return []

    def all(self, offset: int = 0, limit: int = 1000):
        """
        获取所有记录，支持分页
        :param offset: 偏移量，默认0
        :param limit: 限制返回数量，默认1000
        :return: 包含key和value的字典列表
        """
        with self.get_session() as session:
            try:
                result = (
                    session.query(self.model_class).offset(offset).limit(limit).all()
                )
                return [to_dict(item) for item in result]
            except Exception as e:
                print(f"Error: {e}")
                return []

    def find_value(self, value: dict, limit=1000):
        with self.get_session() as session:
            try:
                result = (
                    session.query(self.model_class)
                    .filter(self.model_class.value.contains(value))
                    .limit(limit)
                    .all()
                )
                return [to_dict(item) for item in result]
            except Exception as e:
                print(f"Error: {e}")
                return []

    def find_key(self, key: str, limit=1000):
        with self.get_session() as session:
            try:
                result = (
                    session.query(self.model_class)
                    .filter(self.model_class.key.contains(key))
                    .limit(limit)
                    .all()
                )
                return [to_dict(item) for item in result]
            except Exception as e:
                print(f"Error: {e}")
                return []

    def find(self, key: str = "", value: dict = {}, limit=1000):
        with self.get_session() as session:
            try:
                result = (
                    session.query(self.model_class)
                    .filter(
                        self.model_class.key.contains(key),
                        self.model_class.value.contains(value),
                    )
                    .limit(limit)
                    .all()
                )
                return [to_dict(item) for item in result]
            except Exception as e:
                print(f"Error: {e}")
                return []

    def getMany(self, keys: list[str], offset: int = 0, limit: int = 1000):
        """
        批量获取多个key的值，支持分页
        :param keys: 要查询的key列表
        :param offset: 偏移量，默认0
        :param limit: 限制返回数量，默认1000
        :return: 包含key和value的字典列表
        """
        with self.get_session() as session:
            try:
                result = (
                    session.query(self.model_class)
                    .filter(self.model_class.key.in_(keys))
                    .offset(offset)
                    .limit(limit)
                    .all()
                )
                return [to_dict(item) for item in result]
            except Exception as e:
                print(f"Error: {e}")
                return []

    @staticmethod
    def create_cache_model(table_name):
        return type(
            table_name,
            (Base,),
            {
                "__tablename__": table_name,
                "key": Column(String(255), primary_key=True),
                "value": Column(JSONB),
                "__table_args__": (
                    PrimaryKeyConstraint("key", name=f"{table_name}_pkey"),
                ),
            },
        )
