import json
import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Mapping, Set
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Index, select, func, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import operators

Base = declarative_base()

class KVDatabaseModel(Base):
    """SQLAlchemy模型定义用于KV存储"""
    __tablename__ = 'kv_store'  # 默认表名，可以在初始化时修改
    
    key = Column(String(255), primary_key=True)
    value = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # 索引将在表创建时动态添加

class KVDatabase:
    def __init__(self, connection_string: str, table_name: str = 'kv_store'):
        """
        初始化KV数据库
        
        Args:
            connection_string: PostgreSQL连接字符串，例如："postgresql://user:password@localhost/dbname"
            table_name: 表名，默认为'kv_store'
        """
        self.table_name = table_name
        self.initialized = False
        
        # 动态设置表名
        self.Model = type('KVModel', (KVDatabaseModel,), {'__tablename__': table_name})
        
        # 创建连接
        self.engine = create_engine(
            connection_string,
            pool_size=50,  # 连接池大小
            max_overflow=50,  # 最大溢出连接数
            pool_timeout=30,  # 连接超时时间（秒）
            pool_recycle=7200,  # 连接回收时间（秒）
            pool_pre_ping=True,  # 连接前检查
            echo=False,  # 不输出SQL语句
            connect_args={
                'connect_timeout': 3,
                'options': '-c statement_timeout=15000 -c idle_in_transaction_session_timeout=15000'
            }
        )
        
        self.Session = sessionmaker(bind=self.engine)
    
    def _ensure_initialized(self):
        """确保数据库表已初始化"""
        if not self.initialized:
            # 创建表（如果不存在）
            if not self.engine.dialect.has_table(self.engine, self.table_name):
                self.Model.__table__.create(self.engine, checkfirst=True)
                
                # 创建GIN索引
                with self.engine.connect() as conn:
                    conn.execute(text(
                        f'CREATE INDEX IF NOT EXISTS "IDX_{self.table_name}_value_gin" ON "{self.table_name}" USING gin (value);'
                    ))
            
            self.initialized = True
    
    def put(self, key: str, value: Any) -> None:
        """存储键值对"""
        self._ensure_initialized()
        
        with self.Session() as session:
            record = session.query(self.Model).filter_by(key=key).first()
            if record:
                record.value = value
                record.updated_at = datetime.datetime.utcnow()
            else:
                record = self.Model(key=key, value=value)
                session.add(record)
            session.commit()
    
    def merge(self, key: str, partial_value: dict) -> bool:
        """合并部分值到现有记录"""
        self._ensure_initialized()
        
        with self.Session() as session:
            # 使用原生SQL实现JSONB合并
            result = session.execute(
                text(f"UPDATE {self.table_name} SET value = value || :new_value::jsonb, "
                     f"updated_at = CURRENT_TIMESTAMP WHERE key = :key"),
                {"key": key, "new_value": json.dumps(partial_value)}
            )
            session.commit()
            return result.rowcount > 0
    
    def get(self, key: str, expire: Optional[int] = None) -> Optional[Any]:
        """
        获取键对应的值
        
        Args:
            key: 要查询的键
            expire: 可选的过期时间（秒）
        
        Returns:
            键对应的值，如果不存在则返回None
        """
        self._ensure_initialized()
        
        with self.Session() as session:
            record = session.query(self.Model).filter_by(key=key).first()
            
            if not record:
                return None
            
            # 检查是否过期
            if expire is not None:
                current_time = datetime.datetime.utcnow().timestamp()
                created_time = record.created_at.timestamp()
                if current_time - created_time > expire:
                    # 删除过期数据
                    session.delete(record)
                    session.commit()
                    return None
            
            return record.value
    
    def delete(self, key: str) -> bool:
        """删除键值对"""
        self._ensure_initialized()
        
        with self.Session() as session:
            result = session.query(self.Model).filter_by(key=key).delete()
            session.commit()
            return result > 0
    
    def add(self, key: str, value: Any) -> None:
        """添加新键值对，如果键已存在则抛出错误"""
        self._ensure_initialized()
        
        with self.Session() as session:
            existing = session.query(self.Model).filter_by(key=key).first()
            if existing:
                raise ValueError(f'Key "{key}" already exists')
            
            record = self.Model(key=key, value=value)
            session.add(record)
            session.commit()
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.initialized:
            self.engine.dispose()
            self.initialized = False
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有键值对"""
        self._ensure_initialized()
        
        result = {}
        with self.Session() as session:
            records = session.query(self.Model).all()
            for record in records:
                result[record.key] = record.value
        
        return result
    
    def keys(self) -> List[str]:
        """获取所有键"""
        self._ensure_initialized()
        
        with self.Session() as session:
            records = session.query(self.Model.key).all()
            return [record[0] for record in records]
    
    def has(self, key: str) -> bool:
        """检查键是否存在"""
        self._ensure_initialized()
        
        with self.Session() as session:
            return session.query(self.Model).filter_by(key=key).count() > 0
    
    def put_many(self, entries: List[Tuple[str, Any]], batch_size: int = 1000) -> None:
        """批量添加键值对"""
        self._ensure_initialized()
        
        with self.Session() as session:
            for i in range(0, len(entries), batch_size):
                batch = entries[i:i+batch_size]
                
                # 构建批量插入SQL
                placeholders = []
                params = {}
                
                for idx, (key, value) in enumerate(batch):
                    k_param = f"key_{idx}"
                    v_param = f"value_{idx}"
                    placeholders.append(f"(:{k_param}, :{v_param}::jsonb, NOW(), NOW())")
                    params[k_param] = key
                    params[v_param] = json.dumps(value)
                
                sql = f"""
                INSERT INTO "{self.table_name}" (key, value, created_at, updated_at)
                VALUES {', '.join(placeholders)}
                ON CONFLICT (key) 
                DO UPDATE SET 
                  value = EXCLUDED.value,
                  updated_at = EXCLUDED.updated_at
                """
                
                session.execute(text(sql), params)
            
            session.commit()
    
    def delete_many(self, keys: List[str]) -> int:
        """批量删除键"""
        self._ensure_initialized()
        
        with self.Session() as session:
            result = session.query(self.Model).filter(self.Model.key.in_(keys)).delete(synchronize_session=False)
            session.commit()
            return result
    
    def clear(self) -> None:
        """清空数据库"""
        self._ensure_initialized()
        
        with self.Session() as session:
            session.query(self.Model).delete()
            session.commit()
    
    def count(self) -> int:
        """获取数据库中的记录数量"""
        self._ensure_initialized()
        
        with self.Session() as session:
            return session.query(self.Model).count()
    
    def find_bool_values(self, bool_value: bool, first: bool = True, 
                         order_by: str = 'ASC') -> Union[List[str], Optional[str]]:
        """
        查找布尔值记录
        
        Args:
            bool_value: True 或 False
            first: 是否只返回第一条记录
            order_by: 排序方式 'ASC' 或 'DESC'
        
        Returns:
            如果 first 为 True 返回单个键或 None，否则返回键列表
        """
        self._ensure_initialized()
        
        with self.Session() as session:
            query = session.query(self.Model.key).filter(
                self.Model.value == json.dumps(bool_value)
            )
            
            if order_by.upper() == 'ASC':
                query = query.order_by(self.Model.created_at.asc())
            else:
                query = query.order_by(self.Model.created_at.desc())
            
            if first:
                result = query.first()
                return result[0] if result else None
            else:
                results = query.all()
                return [r[0] for r in results]
    
    def search_json(self, search_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        高级JSON搜索
        
        Args:
            search_options: 搜索选项，包含 contains, limit, cursor 等
        
        Returns:
            包含 data 和 nextCursor 的字典
        """
        self._ensure_initialized()
        
        limit = search_options.get('limit', 100)
        contains = search_options.get('contains')
        cursor = search_options.get('cursor')
        
        with self.Session() as session:
            query = session.query(self.Model)
            
            if contains:
                # 对于PostgreSQL，@>操作符检查左侧JSON是否包含右侧JSON
                query = query.filter(self.Model.value.op('@>')(json.dumps(contains)))
            
            if cursor:
                query = query.filter(self.Model.key > cursor)
            
            query = query.order_by(self.Model.key.asc())
            query = query.limit(limit + 1)
            
            results = query.all()
            
            has_more = len(results) > limit
            data = results[:limit]
            next_cursor = data[-1].key if has_more and data else None
            
            return {
                'data': [{'key': r.key, 'value': r.value} for r in data],
                'nextCursor': next_cursor
            }
    
    def find_by_update_time(self, timestamp: int, first: bool = True,
                            type_: str = 'after', order_by: str = 'ASC') -> Union[List[Dict], Optional[str]]:
        """
        查找更新时间在指定时间前后的记录
        
        Args:
            timestamp: 时间戳（毫秒）
            first: 是否只返回第一条记录
            type_: 'before' 或 'after'
            order_by: 排序方式
        """
        self._ensure_initialized()
        
        dt = datetime.datetime.fromtimestamp(timestamp / 1000.0)
        
        with self.Session() as session:
            query = session.query(self.Model.key, self.Model.value)
            
            if type_ == 'before':
                query = query.filter(self.Model.updated_at < dt)
            else:
                query = query.filter(self.Model.updated_at > dt)
            
            if order_by.upper() == 'ASC':
                query = query.order_by(self.Model.updated_at.asc())
            else:
                query = query.order_by(self.Model.updated_at.desc())
            
            if first:
                result = query.first()
                return result[0] if result else None
            else:
                results = query.all()
                return [{'key': r[0], 'value': r[1]} for r in results]
    
    def search_by_time(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        按时间搜索记录
        
        Args:
            params: 搜索参数，包含 timestamp, take, type, orderBy, timeColumn 等
        """
        self._ensure_initialized()
        
        timestamp = params.get('timestamp')
        take = params.get('take', 1)
        type_ = params.get('type', 'after')
        order_by = params.get('orderBy', 'ASC')
        time_column = params.get('timeColumn', 'updated_at')
        
        dt = datetime.datetime.fromtimestamp(timestamp / 1000.0)
        
        with self.Session() as session:
            query = session.query(self.Model.key, self.Model.value)
            
            # 选择正确的时间列
            column = self.Model.updated_at if time_column == 'updated_at' else self.Model.created_at
            
            # 设置过滤条件
            if type_ == 'before':
                query = query.filter(column < dt)
            else:
                query = query.filter(column > dt)
            
            # 设置排序
            if order_by.upper() == 'ASC':
                query = query.order_by(column.asc())
            else:
                query = query.order_by(column.desc())
            
            # 设置限制
            query = query.limit(take)
            
            results = query.all()
            return [{'key': r[0], 'value': r[1]} for r in results]
    
    def search_json_by_time(self, search_options: Dict[str, Any], 
                          time_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        优化后的JSON和时间复合搜索
        
        Args:
            search_options: JSON搜索选项，包含 contains, equals, path, value 等
            time_options: 时间搜索选项，包含 timestamp, take, type, orderBy, timeColumn 等
        """
        self._ensure_initialized()
        
        # 提取时间参数
        timestamp = time_options.get('timestamp')
        take = time_options.get('take', 1)
        type_ = time_options.get('type', 'after')
        order_by = time_options.get('orderBy', 'ASC')
        time_column = time_options.get('timeColumn', 'updated_at')
        
        dt = datetime.datetime.fromtimestamp(timestamp / 1000.0)
        
        with self.Session() as session:
            query = session.query(self.Model.key, self.Model.value)
            
            # 选择正确的时间列
            column = self.Model.updated_at if time_column == 'updated_at' else self.Model.created_at
            
            # 设置时间过滤条件
            if type_ == 'before':
                query = query.filter(column < dt)
            else:
                query = query.filter(column > dt)
            
            # 添加JSON搜索条件
            contains = search_options.get('contains')
            equals = search_options.get('equals')
            path = search_options.get('path')
            value = search_options.get('value')
            
            if contains:
                query = query.filter(self.Model.value.op('@>')(json.dumps(contains)))
            
            if equals:
                query = query.filter(self.Model.value == json.dumps(equals))
            
            if path is not None and value is not None:
                # 使用原生SQL实现路径访问
                path_expr = f"value #>> '{{{{{}}}}}' = :path_value"
                query = query.filter(text(path_expr)).params(path_value=str(value))
            
            # 设置排序
            if order_by.upper() == 'ASC':
                query = query.order_by(column.asc())
            else:
                query = query.order_by(column.desc())
            
            # 设置限制
            query = query.limit(take)
            
            try:
                results = query.all()
                return [{'key': r[0], 'value': r[1]} for r in results]
            except Exception as e:
                print(f"查询错误: {str(e)}")
                raise
