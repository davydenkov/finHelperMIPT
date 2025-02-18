"""create quotes table

Revision ID: 7ddda8fd22bc
Revises: 
Create Date: 2025-02-18 20:54:57.300037

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7ddda8fd22bc'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    sql = """
    CREATE TABLE quotes (
    id SERIAL PRIMARY KEY,
    figi VARCHAR(20) NOT NULL,
    open_price FLOAT NOT NULL,
    close_price FLOAT NOT NULL,
    high_price FLOAT NOT NULL,
    low_price FLOAT NOT NULL,
    volume INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (figi, timestamp) -- Обязательно для upsert
    )    
    """
    op.execute(sql)
    #op.create_table(
    #    'quotes',
    #    sa.Column('id', sa.Integer, primary_key=True),
    #    sa.Column('figi', sa.String(20), nullable=False),
    #    sa.Column('description', sa.Unicode(200)),
    #)

def downgrade() -> None:
    op.drop_table('quotes')
