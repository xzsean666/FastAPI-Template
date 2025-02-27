import json
import time
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, String, Text, DateTime, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import sqlite3

Base = declarative_base()

# JSON序列化处理器，处理BigInt等特殊类型
def bigint_handler(obj):
    if isinstance(obj, int) and obj > 9007199254740991:  # JS Number.MAX_SAFE_INTEGER
        return str(obj)
    return obj

class KVDatabase:
    def __init__(self, datasource_or_url: Optional[str] = None, table_name: str = "kv_store"):
        self.table_name = table_name
        self.initialized = False
        
        # 创建动态模型类
        class CustomKVStore(Base):
            __tablename__ = table_name
            
            key = Column(String(255), primary_key=True)
            value = Column(Text, nullable=False)
            created_at = Column(DateTime, default=datetime.now)
            updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
            
        self.CustomKVStore = CustomKVStore
        
        # 创建引擎
        db_url = f"sqlite:///{datasource_or_url or ':memory:'}"
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    async def ensure_initialized(self) -> None:
        if not self.initialized:
            # 创建表
            metadata = MetaData()
            
            # 检查表是否存在
            inspector = sa.inspect(self.engine)
            if not inspector.has_table(self.table_name):
                Base.metadata.create_all(self.engine)
            
            self.initialized = True

    async def put(self, key: str, value: Any) -> None:
        await self.ensure_initialized()
        
        with self.Session() as session:
            record = session.query(self.CustomKVStore).filter_by(key=key).first()
            
            if record:
                record.value = json.dumps(value, default=bigint_handler)
                record.updated_at = datetime.now()
            else:
                new_record = self.CustomKVStore(
                    key=key,
                    value=json.dumps(value, default=bigint_handler)
                )
                session.add(new_record)
                
            session.commit()

    async def get(self, key: str, expire: Optional[int] = None) -> Any:
        await self.ensure_initialized()
        
        with self.Session() as session:
            record = session.query(self.CustomKVStore).filter_by(key=key).first()
            
            if not record:
                return None
                
            # 处理过期逻辑
            if expire is not None:
                current_time = int(time.time())
                created_time = int(record.created_at.timestamp())
                
                if current_time - created_time > expire:
                    return None
                    
            return json.loads(record.value)

    async def delete(self, key: str) -> bool:
        await self.ensure_initialized()
        
        with self.Session() as session:
            record = session.query(self.CustomKVStore).filter_by(key=key).first()
            if record:
                session.delete(record)
                session.commit()
                return True
            return False

    async def add(self, key: str, value: Any) -> None:
        await self.ensure_initialized()
        
        with self.Session() as session:
            existing = session.query(self.CustomKVStore).filter_by(key=key).first()
            if existing:
                raise ValueError(f'Key "{key}" already exists')
                
            new_record = self.CustomKVStore(
                key=key,
                value=json.dumps(value, default=bigint_handler)
            )
            session.add(new_record)
            session.commit()

    async def close(self) -> None:
        if self.initialized:
            self.engine.dispose()
            self.initialized = False

    async def get_all(self) -> Dict[str, Any]:
        await self.ensure_initialized()
        
        with self.Session() as session:
            records = session.query(self.CustomKVStore).all()
            return {record.key: json.loads(record.value) for record in records}

    async def get_many(self, limit: int = 10) -> Dict[str, Any]:
        await self.ensure_initialized()
        
        with self.Session() as session:
            records = session.query(self.CustomKVStore).limit(limit).all()
            return {record.key: json.loads(record.value) for record in records}

    async def keys(self) -> List[str]:
        await self.ensure_initialized()
        
        with self.Session() as session:
            records = session.query(self.CustomKVStore.key).all()
            return [record.key for record in records]

    async def has(self, key: str) -> bool:
        await self.ensure_initialized()
        
        with self.Session() as session:
            count = session.query(self.CustomKVStore).filter_by(key=key).count()
            return count > 0

    async def put_many(self, entries: List[Tuple[str, Any]], batch_size: int = 1000) -> None:
        await self.ensure_initialized()
        
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            
            with self.Session() as session:
                for key, value in batch:
                    record = session.query(self.CustomKVStore).filter_by(key=key).first()
                    
                    if record:
                        record.value = json.dumps(value, default=bigint_handler)
                        record.updated_at = datetime.now()
                    else:
                        new_record = self.CustomKVStore(
                            key=key,
                            value=json.dumps(value, default=bigint_handler)
                        )
                        session.add(new_record)
                
                session.commit()

    async def delete_many(self, keys: List[str]) -> int:
        await self.ensure_initialized()
        
        with self.Session() as session:
            count = session.query(self.CustomKVStore).filter(self.CustomKVStore.key.in_(keys)).delete(synchronize_session=False)
            session.commit()
            return count

    async def clear(self) -> None:
        await self.ensure_initialized()
        
        with self.Session() as session:
            session.query(self.CustomKVStore).delete()
            session.commit()

    async def count(self) -> int:
        await self.ensure_initialized()
        
        with self.Session() as session:
            return session.query(self.CustomKVStore).count()

    async def find_by_value(self, value: Any, exact: bool = True) -> List[str]:
        await self.ensure_initialized()
        
        with self.Session() as session:
            if exact:
                value_str = json.dumps(value, default=bigint_handler)
                records = session.query(self.CustomKVStore).filter(self.CustomKVStore.value == value_str).all()
            else:
                search_value = value if isinstance(value, str) else json.dumps(value, default=bigint_handler)
                records = session.query(self.CustomKVStore).filter(self.CustomKVStore.value.like(f'%{search_value}%')).all()
                
            return [record.key for record in records]

    async def find_by_condition(self, condition: Callable[[Any], bool]) -> Dict[str, Any]:
        await self.ensure_initialized()
        
        with self.Session() as session:
            records = session.query(self.CustomKVStore).all()
            result = {}
            
            for record in records:
                value = json.loads(record.value)
                if condition(value):
                    result[record.key] = value
                    
            return result
